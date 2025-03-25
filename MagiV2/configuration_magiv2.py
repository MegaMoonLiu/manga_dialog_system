from transformers import PretrainedConfig, VisionEncoderDecoderConfig
from typing import List


class Magiv2Config(PretrainedConfig):
    model_type = "magiv2"

    def __init__(
        self,
        disable_ocr: bool = False,  # 是否禁用 OCR 功能
        disable_crop_embeddings: bool = False,  # 是否禁用裁剪嵌入功能
        disable_detections: bool = False,  # 是否禁用检测功能
        detection_model_config: dict = None,  # 用于检测模型的配置（字典形式）
        ocr_model_config: dict = None,  #   用于 OCR 模型的配置（字典形式）
        crop_embedding_model_config: dict = None,  # 用于裁剪嵌入模型的配置（字典形式）
        detection_image_preprocessing_config: dict = None,  #    检测模型的图像预处理配置
        ocr_pretrained_processor_path: str = None,  #   OCR 的预训练处理器路径
        crop_embedding_image_preprocessing_config: dict = None,  #   裁剪嵌入的图像预处理配置
        **kwargs,
    ):
        self.disable_ocr = disable_ocr
        self.disable_crop_embeddings = disable_crop_embeddings
        self.disable_detections = disable_detections
        self.kwargs = kwargs
        self.detection_model_config = None
        self.ocr_model_config = None
        self.crop_embedding_model_config = None
        if detection_model_config is not None:
            self.detection_model_config = PretrainedConfig.from_dict(
                detection_model_config
            )
        if ocr_model_config is not None:
            self.ocr_model_config = VisionEncoderDecoderConfig.from_dict(
                ocr_model_config
            )
        if crop_embedding_model_config is not None:
            self.crop_embedding_model_config = PretrainedConfig.from_dict(
                crop_embedding_model_config
            )

        self.detection_image_preprocessing_config = detection_image_preprocessing_config
        self.ocr_pretrained_processor_path = ocr_pretrained_processor_path
        self.crop_embedding_image_preprocessing_config = (
            crop_embedding_image_preprocessing_config
        )
        super().__init__(**kwargs)
