import json
from transformers import (
    PreTrainedModel,
    VisionEncoderDecoderModel,  #  用于从图像中提取文本内容
    ViTMAEModel,  #   用于计算裁剪区域（如角色图像）的特征嵌入
    ConditionalDetrModel,  #   用于检测目标的边界框（Bounding Box）及其分类
)
from transformers.models.conditional_detr.modeling_conditional_detr import (
    ConditionalDetrMLPPredictionHead,
    ConditionalDetrModelOutput,
    ConditionalDetrHungarianMatcher,
    inverse_sigmoid,
)
from .configuration_magiv2 import Magiv2Config
from .processing_magiv2 import Magiv2Processor
from torch import nn
from typing import Optional, List
import torch
from einops import rearrange, repeat
from .utils import (
    move_to_device,
    visualise_single_image_prediction,
    sort_panels,
    sort_text_boxes_in_reading_order,
)
from transformers.image_transforms import center_to_corners_format
from .utils import UnionFind, sort_panels, sort_text_boxes_in_reading_order
import pulp
import scipy
import numpy as np

import PIL.Image
import os
import re
import cv2
from manga_ocr import MangaOcr

mocr = MangaOcr()
text_path = "/export/users/liu/Manga_Whisperer/"


class Magiv2Model(PreTrainedModel):
    config_class = Magiv2Config
    page = 0

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.processor = Magiv2Processor(config)
        if not config.disable_ocr:
            self.ocr_model = VisionEncoderDecoderModel(config.ocr_model_config)
        if not config.disable_crop_embeddings:
            self.crop_embedding_model = ViTMAEModel(config.crop_embedding_model_config)
        if not config.disable_detections:
            self.num_non_obj_tokens = 5
            self.detection_transformer = ConditionalDetrModel(
                config.detection_model_config
            )
            self.bbox_predictor = ConditionalDetrMLPPredictionHead(
                input_dim=config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=4,
                num_layers=3,
            )
            self.character_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=3 * config.detection_model_config.d_model
                + (
                    2 * config.crop_embedding_model_config.hidden_size
                    if not config.disable_crop_embeddings
                    else 0
                ),
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1,
                num_layers=3,
            )
            self.text_character_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=3 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1,
                num_layers=3,
            )
            self.text_tail_matching_head = ConditionalDetrMLPPredictionHead(
                input_dim=2 * config.detection_model_config.d_model,
                hidden_dim=config.detection_model_config.d_model,
                output_dim=1,
                num_layers=3,
            )
            self.class_labels_classifier = nn.Linear(
                config.detection_model_config.d_model,
                config.detection_model_config.num_labels,
            )
            self.is_this_text_a_dialogue = nn.Linear(
                config.detection_model_config.d_model, 1
            )
            self.matcher = ConditionalDetrHungarianMatcher(
                class_cost=config.detection_model_config.class_cost,
                bbox_cost=config.detection_model_config.bbox_cost,
                giou_cost=config.detection_model_config.giou_cost,
            )

    def move_to_device(self, input):
        return move_to_device(input, self.device)

    @torch.no_grad()
    # 处理一个完整的章节或多页图像，进行以下任务：
    # 提取 OCR 文本。
    # 检测角色并聚类。
    # 在页间关联角色和文本。
    def do_chapter_wide_prediction(
        self,
        pages_in_order,
        character_bank,
        eta=0.75,
        batch_size=8,
        use_tqdm=False,
        do_ocr=True,
    ):
        texts = []
        characters = []
        character_clusters = []
        if use_tqdm:
            from tqdm import tqdm

            iterator = tqdm(range(0, len(pages_in_order), batch_size))
        else:
            iterator = range(0, len(pages_in_order), batch_size)
        per_page_results = []
        for i in iterator:
            pages = pages_in_order[i : i + batch_size]
            results = self.predict_detections_and_associations(pages)
            per_page_results.extend([result for result in results])

        texts = [result["texts"] for result in per_page_results]
        characters = [result["characters"] for result in per_page_results]
        character_clusters = [
            result["character_cluster_labels"] for result in per_page_results
        ]
        assigned_character_names = self.assign_names_to_characters(
            pages_in_order, characters, character_bank, character_clusters, eta=eta
        )
        if do_ocr:
            ocr = self.predict_ocr(pages_in_order, texts, use_tqdm=use_tqdm)
        offset_characters = 0
        iteration_over = zip(per_page_results, ocr) if do_ocr else per_page_results
        for iter in iteration_over:
            if do_ocr:
                result, ocr_for_page = iter
                result["ocr"] = ocr_for_page
            else:
                result = iter
            result["character_names"] = assigned_character_names[
                offset_characters : offset_characters + len(result["characters"])
            ]
            offset_characters += len(result["characters"])
        return per_page_results

    # 为检测到的角色分配名称
    def assign_names_to_characters(
        self, images, character_bboxes, character_bank, character_clusters, eta=0.75
    ):
        if len(character_bank["images"]) == 0:
            return [
                "Other"
                for bboxes_for_image in character_bboxes
                for bbox in bboxes_for_image
            ]
        chapter_wide_char_embeddings = self.predict_crop_embeddings(
            images, character_bboxes
        )
        chapter_wide_char_embeddings = torch.cat(chapter_wide_char_embeddings, dim=0)
        chapter_wide_char_embeddings = (
            torch.nn.functional.normalize(chapter_wide_char_embeddings, p=2, dim=1)
            .cpu()
            .numpy()
        )
        # create must-link and cannot link constraints from character_clusters
        must_link = []
        cannot_link = []
        offset = 0
        for clusters_per_image in character_clusters:
            for i in range(len(clusters_per_image)):
                for j in range(i + 1, len(clusters_per_image)):
                    if clusters_per_image[i] == clusters_per_image[j]:
                        must_link.append((offset + i, offset + j))
                    else:
                        cannot_link.append((offset + i, offset + j))
            offset += len(clusters_per_image)
        character_bank_for_this_chapter = self.predict_crop_embeddings(
            character_bank["images"],
            [[[0, 0, x.shape[1], x.shape[0]]] for x in character_bank["images"]],
        )
        character_bank_for_this_chapter = torch.cat(
            character_bank_for_this_chapter, dim=0
        )
        character_bank_for_this_chapter = (
            torch.nn.functional.normalize(character_bank_for_this_chapter, p=2, dim=1)
            .cpu()
            .numpy()
        )
        costs = scipy.spatial.distance.cdist(
            chapter_wide_char_embeddings, character_bank_for_this_chapter
        )
        none_of_the_above = eta * np.ones((costs.shape[0], 1))
        costs = np.concatenate([costs, none_of_the_above], axis=1)
        sense = pulp.LpMinimize
        num_supply, num_demand = costs.shape
        problem = pulp.LpProblem("Optimal_Transport_Problem", sense)
        x = pulp.LpVariable.dicts(
            "x",
            ((i, j) for i in range(num_supply) for j in range(num_demand)),
            cat="Binary",
        )
        # Objective Function to minimize
        problem += pulp.lpSum(
            [
                costs[i][j] * x[(i, j)]
                for i in range(num_supply)
                for j in range(num_demand)
            ]
        )
        # each crop must be assigned to exactly one character
        for i in range(num_supply):
            problem += (
                pulp.lpSum([x[(i, j)] for j in range(num_demand)]) == 1,
                f"Supply_{i}_Total_Assignment",
            )
        # cannot link constraints
        for j in range(num_demand - 1):
            for s1, s2 in cannot_link:
                problem += (
                    x[(s1, j)] + x[(s2, j)] <= 1,
                    f"Exclusion_{s1}_{s2}_Demand_{j}",
                )
        # must link constraints
        for j in range(num_demand):
            for s1, s2 in must_link:
                problem += (
                    x[(s1, j)] - x[(s2, j)] == 0,
                    f"Inclusion_{s1}_{s2}_Demand_{j}",
                )
        problem.solve()
        assignments = []
        for v in problem.variables():
            if v.varValue > 0:
                index, assignment = v.name.split("(")[1].split(")")[0].split(",")
                assignment = assignment[1:]
                assignments.append((int(index), int(assignment)))

        labels = np.zeros(num_supply)
        for i, j in assignments:
            labels[i] = j

        return [
            (
                character_bank["names"][int(i)]
                if i < len(character_bank["names"])
                else "Other"
            )
            for i in labels
        ]

    # 检测与关联
    #   输入：多张图片
    #   输出：
    #   检测到的边界框（如文本框、角色框）
    #   文本和角色或对话框之间的关联关系
    #   改
    def predict_detections_and_associations(
        self,
        images,
        move_to_device_fn=None,
        character_detection_threshold=0.3,
        panel_detection_threshold=0.2,
        text_detection_threshold=0.3,
        tail_detection_threshold=0.34,
        character_character_matching_threshold=0.65,
        text_character_matching_threshold=0.35,
        text_tail_matching_threshold=0.3,
        text_classification_threshold=0.5,
    ):
        assert not self.config.disable_detections
        move_to_device_fn = (
            self.move_to_device if move_to_device_fn is None else move_to_device_fn
        )

        inputs_to_detection_transformer = (
            self.processor.preprocess_inputs_for_detection(images)
        )
        inputs_to_detection_transformer = move_to_device_fn(
            inputs_to_detection_transformer
        )

        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer
        )
        predicted_class_scores, predicted_bboxes = (
            self._get_predicted_bboxes_and_classes(detection_transformer_output)
        )

        original_image_sizes = torch.stack(
            [torch.tensor(img.shape[:2]) for img in images], dim=0
        ).to(predicted_bboxes.device)

        batch_scores, batch_labels = predicted_class_scores.max(-1)
        batch_scores = batch_scores.sigmoid()
        batch_labels = batch_labels.long()
        batch_bboxes = center_to_corners_format(predicted_bboxes)

        # scale the bboxes back to the original image size
        if isinstance(original_image_sizes, List):
            img_h = torch.Tensor([i[0] for i in original_image_sizes])
            img_w = torch.Tensor([i[1] for i in original_image_sizes])
        else:
            img_h, img_w = original_image_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
            batch_bboxes.device
        )
        batch_bboxes = batch_bboxes * scale_fct[:, None, :]

        batch_panel_indices = self.processor._get_indices_of_panels_to_keep(
            batch_scores, batch_labels, batch_bboxes, panel_detection_threshold
        )
        batch_character_indices = self.processor._get_indices_of_characters_to_keep(
            batch_scores, batch_labels, batch_bboxes, character_detection_threshold
        )
        batch_text_indices = self.processor._get_indices_of_texts_to_keep(
            batch_scores, batch_labels, batch_bboxes, text_detection_threshold
        )
        batch_tail_indices = self.processor._get_indices_of_tails_to_keep(
            batch_scores, batch_labels, batch_bboxes, tail_detection_threshold
        )

        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
            detection_transformer_output
        )
        predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(
            detection_transformer_output
        )
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
            detection_transformer_output
        )

        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=[
                x[i]
                for x, i in zip(predicted_obj_tokens_for_batch, batch_character_indices)
            ],
            text_obj_tokens_for_this_batch=[
                x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_text_indices)
            ],
            t2c_tokens_for_batch=predicted_t2c_tokens_for_batch,
            apply_sigmoid=True,
        )

        character_bboxes_in_batch = [
            batch_bboxes[i][j] for i, j in enumerate(batch_character_indices)
        ]
        character_character_affinity_matrices = (
            self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=[
                    x[i]
                    for x, i in zip(
                        predicted_obj_tokens_for_batch, batch_character_indices
                    )
                ],
                crop_embeddings_for_batch=self.predict_crop_embeddings(
                    images, character_bboxes_in_batch, move_to_device_fn
                ),
                c2c_tokens_for_batch=predicted_c2c_tokens_for_batch,
                apply_sigmoid=True,
            )
        )

        text_tail_affinity_matrices = self._get_text_tail_affinity_matrices(
            text_obj_tokens_for_this_batch=[
                x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_text_indices)
            ],
            tail_obj_tokens_for_batch=[
                x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_tail_indices)
            ],
            apply_sigmoid=True,
        )

        is_this_text_a_dialogue = self._get_text_classification(
            [x[i] for x, i in zip(predicted_obj_tokens_for_batch, batch_text_indices)]
        )

        results = []
        is_essential_text_results = []
        for batch_index in range(len(batch_scores)):
            panel_indices = batch_panel_indices[batch_index]
            character_indices = batch_character_indices[batch_index]
            text_indices = batch_text_indices[batch_index]
            tail_indices = batch_tail_indices[batch_index]

            character_bboxes = batch_bboxes[batch_index][character_indices]
            panel_bboxes = batch_bboxes[batch_index][panel_indices]
            text_bboxes = batch_bboxes[batch_index][text_indices]
            tail_bboxes = batch_bboxes[batch_index][tail_indices]

            local_sorted_panel_indices = sort_panels(panel_bboxes)
            panel_bboxes = panel_bboxes[local_sorted_panel_indices]
            local_sorted_text_indices = sort_text_boxes_in_reading_order(
                text_bboxes, panel_bboxes
            )
            text_bboxes = text_bboxes[local_sorted_text_indices]

            character_character_matching_scores = character_character_affinity_matrices[
                batch_index
            ]
            text_character_matching_scores = text_character_affinity_matrices[
                batch_index
            ][local_sorted_text_indices]
            text_tail_matching_scores = text_tail_affinity_matrices[batch_index][
                local_sorted_text_indices
            ]

            # 存储了每段文本的对话标记结果，格式为布尔列表。
            is_essential_text = (
                is_this_text_a_dialogue[batch_index][local_sorted_text_indices]
                > text_classification_threshold
            )
            character_cluster_labels = UnionFind.from_adj_matrix(
                character_character_matching_scores
                > character_character_matching_threshold
            ).get_labels_for_connected_components()

            if 0 in text_character_matching_scores.shape:
                text_character_associations = torch.zeros((0, 2), dtype=torch.long)
            else:
                most_likely_speaker_for_each_text = torch.argmax(
                    text_character_matching_scores, dim=1
                )
                text_indices = torch.arange(len(text_bboxes)).type_as(
                    most_likely_speaker_for_each_text
                )
                text_character_associations = torch.stack(
                    [text_indices, most_likely_speaker_for_each_text], dim=1
                )
                to_keep = (
                    text_character_matching_scores.max(dim=1).values
                    > text_character_matching_threshold
                )
                text_character_associations = text_character_associations[to_keep]

            if 0 in text_tail_matching_scores.shape:
                text_tail_associations = torch.zeros((0, 2), dtype=torch.long)
            else:
                most_likely_tail_for_each_text = torch.argmax(
                    text_tail_matching_scores, dim=1
                )
                text_indices = torch.arange(len(text_bboxes)).type_as(
                    most_likely_tail_for_each_text
                )
                text_tail_associations = torch.stack(
                    [text_indices, most_likely_tail_for_each_text], dim=1
                )
                to_keep = (
                    text_tail_matching_scores.max(dim=1).values
                    > text_tail_matching_threshold
                )
                text_tail_associations = text_tail_associations[to_keep]

            results.append(
                {
                    "panels": panel_bboxes.tolist(),
                    "texts": text_bboxes.tolist(),
                    "characters": character_bboxes.tolist(),
                    "tails": tail_bboxes.tolist(),
                    "text_character_associations": text_character_associations.tolist(),
                    "text_tail_associations": text_tail_associations.tolist(),
                    "character_cluster_labels": character_cluster_labels,
                    "is_essential_text": is_essential_text.tolist(),
                }
            )

            is_essential_text_results.append(
                {f"{self.page}": is_essential_text.tolist()}
            )
            self.page += 1

        with open(text_path + f"results.json", "a", encoding="utf-8") as f:
            json.dump(
                is_essential_text_results,
                f,
                ensure_ascii=False,
                indent=4,
            )

        return results

    # 计算相似度矩阵
    # 它通过模型的检测结果和标注信息（annotations），生成这些相似度矩阵，用于描述输入图像中各个对象的关联关系
    def get_affinity_matrices_given_annotations(
        self, images, annotations, move_to_device_fn=None, apply_sigmoid=True
    ):
        assert not self.config.disable_detections
        move_to_device_fn = (
            self.move_to_device if move_to_device_fn is None else move_to_device_fn
        )

        character_bboxes_in_batch = [
            [
                bbox
                for bbox, label in zip(a["bboxes_as_x1y1x2y2"], a["labels"])
                if label == 0
            ]
            for a in annotations
        ]
        crop_embeddings_for_batch = self.predict_crop_embeddings(
            images, character_bboxes_in_batch, move_to_device_fn
        )

        inputs_to_detection_transformer = (
            self.processor.preprocess_inputs_for_detection(images, annotations)
        )
        inputs_to_detection_transformer = move_to_device_fn(
            inputs_to_detection_transformer
        )
        processed_targets = inputs_to_detection_transformer.pop("labels")

        detection_transformer_output = self._get_detection_transformer_output(
            **inputs_to_detection_transformer
        )
        predicted_obj_tokens_for_batch = self._get_predicted_obj_tokens(
            detection_transformer_output
        )
        predicted_t2c_tokens_for_batch = self._get_predicted_t2c_tokens(
            detection_transformer_output
        )
        predicted_c2c_tokens_for_batch = self._get_predicted_c2c_tokens(
            detection_transformer_output
        )

        predicted_class_scores, predicted_bboxes = (
            self._get_predicted_bboxes_and_classes(detection_transformer_output)
        )
        matching_dict = {
            "logits": predicted_class_scores,
            "pred_boxes": predicted_bboxes,
        }
        indices = self.matcher(matching_dict, processed_targets)

        matched_char_obj_tokens_for_batch = []
        matched_text_obj_tokens_for_batch = []
        matched_tail_obj_tokens_for_batch = []
        t2c_tokens_for_batch = []
        c2c_tokens_for_batch = []

        for j, (pred_idx, tgt_idx) in enumerate(indices):
            target_idx_to_pred_idx = {
                tgt.item(): pred.item() for pred, tgt in zip(pred_idx, tgt_idx)
            }
            targets_for_this_image = processed_targets[j]
            indices_of_text_boxes_in_annotation = [
                i
                for i, label in enumerate(targets_for_this_image["class_labels"])
                if label == 1
            ]
            indices_of_char_boxes_in_annotation = [
                i
                for i, label in enumerate(targets_for_this_image["class_labels"])
                if label == 0
            ]
            indices_of_tail_boxes_in_annotation = [
                i
                for i, label in enumerate(targets_for_this_image["class_labels"])
                if label == 3
            ]
            predicted_text_indices = [
                target_idx_to_pred_idx[i] for i in indices_of_text_boxes_in_annotation
            ]
            predicted_char_indices = [
                target_idx_to_pred_idx[i] for i in indices_of_char_boxes_in_annotation
            ]
            predicted_tail_indices = [
                target_idx_to_pred_idx[i] for i in indices_of_tail_boxes_in_annotation
            ]
            matched_char_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_char_indices]
            )
            matched_text_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_text_indices]
            )
            matched_tail_obj_tokens_for_batch.append(
                predicted_obj_tokens_for_batch[j][predicted_tail_indices]
            )
            t2c_tokens_for_batch.append(predicted_t2c_tokens_for_batch[j])
            c2c_tokens_for_batch.append(predicted_c2c_tokens_for_batch[j])

        text_character_affinity_matrices = self._get_text_character_affinity_matrices(
            character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
            text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            t2c_tokens_for_batch=t2c_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        character_character_affinity_matrices = (
            self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
                crop_embeddings_for_batch=crop_embeddings_for_batch,
                c2c_tokens_for_batch=c2c_tokens_for_batch,
                apply_sigmoid=apply_sigmoid,
            )
        )

        character_character_affinity_matrices_crop_only = (
            self._get_character_character_affinity_matrices(
                character_obj_tokens_for_batch=matched_char_obj_tokens_for_batch,
                crop_embeddings_for_batch=crop_embeddings_for_batch,
                c2c_tokens_for_batch=c2c_tokens_for_batch,
                crop_only=True,
                apply_sigmoid=apply_sigmoid,
            )
        )

        text_tail_affinity_matrices = self._get_text_tail_affinity_matrices(
            text_obj_tokens_for_this_batch=matched_text_obj_tokens_for_batch,
            tail_obj_tokens_for_batch=matched_tail_obj_tokens_for_batch,
            apply_sigmoid=apply_sigmoid,
        )

        is_this_text_a_dialogue = self._get_text_classification(
            matched_text_obj_tokens_for_batch, apply_sigmoid=apply_sigmoid
        )

        return {
            "text_character_affinity_matrices": text_character_affinity_matrices,
            "character_character_affinity_matrices": character_character_affinity_matrices,
            "character_character_affinity_matrices_crop_only": character_character_affinity_matrices_crop_only,
            "text_tail_affinity_matrices": text_tail_affinity_matrices,
            "is_this_text_a_dialogue": is_this_text_a_dialogue,
        }

    # 为裁剪的图像区域生成特征嵌入，帮助模型进行角色识别和特征比较
    def predict_crop_embeddings(
        self,
        images,
        crop_bboxes,
        move_to_device_fn=None,
        mask_ratio=0.0,
        batch_size=256,
    ):
        if self.config.disable_crop_embeddings:
            return None

        assert isinstance(
            crop_bboxes, List
        ), "please provide a list of bboxes for each image to get embeddings for"

        move_to_device_fn = (
            self.move_to_device if move_to_device_fn is None else move_to_device_fn
        )

        # temporarily change the mask ratio from default to the one specified
        old_mask_ratio = self.crop_embedding_model.embeddings.config.mask_ratio
        self.crop_embedding_model.embeddings.config.mask_ratio = mask_ratio

        crops_per_image = []
        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]
        for image, bboxes, num_crops in zip(images, crop_bboxes, num_crops_per_batch):
            crops = self.processor.crop_image(image, bboxes)
            assert len(crops) == num_crops
            crops_per_image.extend(crops)

        if len(crops_per_image) == 0:
            return [
                move_to_device_fn(
                    torch.zeros(0, self.config.crop_embedding_model_config.hidden_size)
                )
                for _ in crop_bboxes
            ]

        crops_per_image = self.processor.preprocess_inputs_for_crop_embeddings(
            crops_per_image
        )
        crops_per_image = move_to_device_fn(crops_per_image)

        # process the crops in batches to avoid OOM
        embeddings = []
        for i in range(0, len(crops_per_image), batch_size):
            crops = crops_per_image[i : i + batch_size]
            embeddings_per_batch = self.crop_embedding_model(crops).last_hidden_state[
                :, 0
            ]
            embeddings.append(embeddings_per_batch)
        embeddings = torch.cat(embeddings, dim=0)

        crop_embeddings_for_batch = []
        for num_crops in num_crops_per_batch:
            crop_embeddings_for_batch.append(embeddings[:num_crops])
            embeddings = embeddings[num_crops:]

        # restore the mask ratio to the default
        self.crop_embedding_model.embeddings.config.mask_ratio = old_mask_ratio

        return crop_embeddings_for_batch

    # 利用 Transformer 模型生成图像区域的文本内容
    #  改
    def predict_ocr(
        self,
        images,  #   图像列表
        crop_bboxes,  #   裁剪后的区域
        move_to_device_fn=None,
        use_tqdm=False,
        batch_size=32,
        max_new_tokens=64,
        save_dir="crops_output",
    ):
        # 确保使用OCR模块
        assert not self.config.disable_ocr

        # 数据转移到GPU
        move_to_device_fn = (
            self.move_to_device if move_to_device_fn is None else move_to_device_fn
        )

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 用于存储裁剪后的区域
        crops_per_image = []

        num_crops_per_batch = [len(bboxes) for bboxes in crop_bboxes]

        # 裁剪图像区域
        all_generated_texts_mocr = []

        for img_idx, (image, bboxes, num_crops) in enumerate(
            zip(images, crop_bboxes, num_crops_per_batch)
        ):
            crops = self.processor.crop_image(image, bboxes)
            assert len(crops) == num_crops
            crops_per_image.extend(crops)

            # 存储裁剪后的图片
            for crop_idx, crop in enumerate(crops):
                # 如果 crop 是张量或 NumPy 数组，需要转换为 BGR 格式保存
                if isinstance(crop, torch.Tensor):
                    crop = crop.permute(1, 2, 0).cpu().numpy()  # [H, W, C]

                crop = (crop * 255).astype("uint8")  # 归一化到 0-255 范围（如需要）
                save_path = os.path.join(
                    save_dir, f"image_{img_idx}_crop_{crop_idx}.png"
                )
                cv2.imwrite(save_path, crop)

        # 空裁剪处理
        if len(crops_per_image) == 0:
            return [[] for _ in crop_bboxes]

        # 预处理裁剪区域
        crops_per_image = self.processor.preprocess_inputs_for_ocr(crops_per_image)
        crops_per_image = move_to_device_fn(crops_per_image)

        # process the crops in batches to avoid OOM
        # 批量处理 OCR
        all_generated_texts = []

        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(range(0, len(crops_per_image), batch_size))
        else:
            pbar = range(0, len(crops_per_image), batch_size)
        for i in pbar:
            crops = crops_per_image[i : i + batch_size]
            generated_ids = self.ocr_model.generate(
                crops, max_new_tokens=max_new_tokens
            )
            generated_texts = self.processor.postprocess_ocr_tokens(generated_ids)
            all_generated_texts.extend(generated_texts)

            # img_mocr = PIL.Image.open(save_path)
            # generated_texts_mocr = mocr(img_mocr)
        texts_for_images = []
        # 对生成的文本进行排序
        for num_crops in num_crops_per_batch:
            texts_for_images.append(
                [x.replace("\n", "") for x in all_generated_texts[:num_crops]]
            )
            all_generated_texts = all_generated_texts[num_crops:]
        return texts_for_images

    # 该函数用于可视化单张图像的预测结果，将预测的内容（如边界框、类别信息等）绘制在图像上，并保存为文件（可选）。
    def visualise_single_image_prediction(
        self, image_as_np_array, predictions, filename=None
    ):
        return visualise_single_image_prediction(
            image_as_np_array, predictions, filename
        )

    @torch.no_grad()

    # 从输入图像的像素值（pixel_values）和可选的像素掩码（pixel_mask）中，调用检测 Transformer 模型，获取其输出结果。
    # 输出（如隐藏状态、预测分数、参考点等）
    def _get_detection_transformer_output(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
    ):
        if self.config.disable_detections:
            raise ValueError(
                "Detection model is disabled. Set disable_detections=False in the config."
            )
        return self.detection_transformer(
            pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True
        )

    # 从检测模型的输出中提取 对象特征向量（object tokens），这些特征向量表示检测到的物体或实体。
    def _get_predicted_obj_tokens(
        self, detection_transformer_output: ConditionalDetrModelOutput
    ):
        return detection_transformer_output.last_hidden_state[
            :, : -self.num_non_obj_tokens
        ]

    # 从检测模型的输出中提取 角色到Character-to-Character Matching, （C2C）特征向量，这些向量用于计算角色之间的关联关系。
    def _get_predicted_c2c_tokens(
        self, detection_transformer_output: ConditionalDetrModelOutput
    ):
        return detection_transformer_output.last_hidden_state[
            :, -self.num_non_obj_tokens
        ]

    # 从检测模型的输出 (ConditionalDetrModelOutput) 中提取 Text-to-Character Matching（T2C）的特征向量
    def _get_predicted_t2c_tokens(
        self, detection_transformer_output: ConditionalDetrModelOutput
    ):
        return detection_transformer_output.last_hidden_state[
            :, -self.num_non_obj_tokens + 1
        ]

    # 从检测模型 (ConditionalDetrModel) 的输出中提取预测的边界框（bounding boxes）和类别分数（class scores）
    def _get_predicted_bboxes_and_classes(
        self,
        detection_transformer_output: ConditionalDetrModelOutput,
    ):
        if self.config.disable_detections:
            raise ValueError(
                "Detection model is disabled. Set disable_detections=False in the config."
            )

        obj = self._get_predicted_obj_tokens(detection_transformer_output)

        predicted_class_scores = self.class_labels_classifier(obj)
        reference = detection_transformer_output.reference_points[
            : -self.num_non_obj_tokens
        ]
        reference_before_sigmoid = inverse_sigmoid(reference).transpose(0, 1)
        predicted_boxes = self.bbox_predictor(obj)
        predicted_boxes[..., :2] += reference_before_sigmoid
        predicted_boxes = predicted_boxes.sigmoid()

        return predicted_class_scores, predicted_boxes

    # 判断这些文本是否为对话
    def _get_text_classification(
        self,
        text_obj_tokens_for_batch: List[torch.FloatTensor],
        apply_sigmoid=False,
    ):
        assert not self.config.disable_detections
        is_this_text_a_dialogue = []
        for text_obj_tokens in text_obj_tokens_for_batch:
            if text_obj_tokens.shape[0] == 0:
                is_this_text_a_dialogue.append(torch.tensor([], dtype=torch.bool))
                continue
            classification = self.is_this_text_a_dialogue(text_obj_tokens).squeeze(-1)
            if apply_sigmoid:
                classification = classification.sigmoid()
            is_this_text_a_dialogue.append(classification)

        return is_this_text_a_dialogue

    # 相似度矩阵    计算角色之间的相似性
    def _get_character_character_affinity_matrices(
        self,
        character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
        crop_embeddings_for_batch: List[torch.FloatTensor] = None,
        c2c_tokens_for_batch: List[torch.FloatTensor] = None,
        crop_only=False,
        apply_sigmoid=True,
    ):
        assert self.config.disable_detections or (
            character_obj_tokens_for_batch is not None
            and c2c_tokens_for_batch is not None
        )
        assert (
            self.config.disable_crop_embeddings or crop_embeddings_for_batch is not None
        )
        assert (
            not self.config.disable_detections
            or not self.config.disable_crop_embeddings
        )

        if crop_only:
            affinity_matrices = []
            for crop_embeddings in crop_embeddings_for_batch:
                crop_embeddings = crop_embeddings / crop_embeddings.norm(
                    dim=-1, keepdim=True
                )
                affinity_matrix = crop_embeddings @ crop_embeddings.T
                affinity_matrices.append(affinity_matrix)
            return affinity_matrices
        affinity_matrices = []
        for batch_index, (character_obj_tokens, c2c) in enumerate(
            zip(character_obj_tokens_for_batch, c2c_tokens_for_batch)
        ):
            if character_obj_tokens.shape[0] == 0:
                affinity_matrices.append(
                    torch.zeros(0, 0).type_as(character_obj_tokens)
                )
                continue
            if not self.config.disable_crop_embeddings:
                crop_embeddings = crop_embeddings_for_batch[batch_index]
                assert character_obj_tokens.shape[0] == crop_embeddings.shape[0]
                character_obj_tokens = torch.cat(
                    [character_obj_tokens, crop_embeddings], dim=-1
                )
            char_i = repeat(
                character_obj_tokens,
                "i d -> i repeat d",
                repeat=character_obj_tokens.shape[0],
            )
            char_j = repeat(
                character_obj_tokens,
                "j d -> repeat j d",
                repeat=character_obj_tokens.shape[0],
            )
            char_ij = rearrange([char_i, char_j], "two i j d -> (i j) (two d)")
            c2c = repeat(c2c, "d -> repeat d", repeat=char_ij.shape[0])
            char_ij_c2c = torch.cat([char_ij, c2c], dim=-1)
            character_character_affinities = self.character_character_matching_head(
                char_ij_c2c
            )
            character_character_affinities = rearrange(
                character_character_affinities, "(i j) 1 -> i j", i=char_i.shape[0]
            )
            character_character_affinities = (
                character_character_affinities + character_character_affinities.T
            ) / 2
            if apply_sigmoid:
                character_character_affinities = (
                    character_character_affinities.sigmoid()
                )
            affinity_matrices.append(character_character_affinities)
        return affinity_matrices

    # 相似度矩阵    计算文本与角色之间的匹配分数
    def _get_text_character_affinity_matrices(
        self,
        character_obj_tokens_for_batch: List[torch.FloatTensor] = None,
        text_obj_tokens_for_this_batch: List[torch.FloatTensor] = None,
        t2c_tokens_for_batch: List[torch.FloatTensor] = None,
        apply_sigmoid=True,
    ):
        assert not self.config.disable_detections
        assert (
            character_obj_tokens_for_batch is not None
            and text_obj_tokens_for_this_batch is not None
            and t2c_tokens_for_batch is not None
        )
        affinity_matrices = []
        for character_obj_tokens, text_obj_tokens, t2c in zip(
            character_obj_tokens_for_batch,
            text_obj_tokens_for_this_batch,
            t2c_tokens_for_batch,
        ):
            if character_obj_tokens.shape[0] == 0 or text_obj_tokens.shape[0] == 0:
                affinity_matrices.append(
                    torch.zeros(
                        text_obj_tokens.shape[0], character_obj_tokens.shape[0]
                    ).type_as(character_obj_tokens)
                )
                continue
            text_i = repeat(
                text_obj_tokens,
                "i d -> i repeat d",
                repeat=character_obj_tokens.shape[0],
            )
            char_j = repeat(
                character_obj_tokens,
                "j d -> repeat j d",
                repeat=text_obj_tokens.shape[0],
            )
            text_char = rearrange([text_i, char_j], "two i j d -> (i j) (two d)")
            t2c = repeat(t2c, "d -> repeat d", repeat=text_char.shape[0])
            text_char_t2c = torch.cat([text_char, t2c], dim=-1)
            text_character_affinities = self.text_character_matching_head(text_char_t2c)
            text_character_affinities = rearrange(
                text_character_affinities, "(i j) 1 -> i j", i=text_i.shape[0]
            )
            if apply_sigmoid:
                text_character_affinities = text_character_affinities.sigmoid()
            affinity_matrices.append(text_character_affinities)
        return affinity_matrices

    # 相似度矩阵    计算文本与尾部框之间的匹配分数
    def _get_text_tail_affinity_matrices(
        self,
        text_obj_tokens_for_this_batch: List[torch.FloatTensor] = None,
        tail_obj_tokens_for_batch: List[torch.FloatTensor] = None,
        apply_sigmoid=True,
    ):
        assert not self.config.disable_detections
        assert (
            tail_obj_tokens_for_batch is not None
            and text_obj_tokens_for_this_batch is not None
        )
        affinity_matrices = []
        for tail_obj_tokens, text_obj_tokens in zip(
            tail_obj_tokens_for_batch, text_obj_tokens_for_this_batch
        ):
            if tail_obj_tokens.shape[0] == 0 or text_obj_tokens.shape[0] == 0:
                affinity_matrices.append(
                    torch.zeros(
                        text_obj_tokens.shape[0], tail_obj_tokens.shape[0]
                    ).type_as(tail_obj_tokens)
                )
                continue
            text_i = repeat(
                text_obj_tokens, "i d -> i repeat d", repeat=tail_obj_tokens.shape[0]
            )
            tail_j = repeat(
                tail_obj_tokens, "j d -> repeat j d", repeat=text_obj_tokens.shape[0]
            )
            text_tail = rearrange([text_i, tail_j], "two i j d -> (i j) (two d)")
            text_tail_affinities = self.text_tail_matching_head(text_tail)
            text_tail_affinities = rearrange(
                text_tail_affinities, "(i j) 1 -> i j", i=text_i.shape[0]
            )
            if apply_sigmoid:
                text_tail_affinities = text_tail_affinities.sigmoid()
            affinity_matrices.append(text_tail_affinities)
        return affinity_matrices
