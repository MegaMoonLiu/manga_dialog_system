import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from janome.tokenizer import Tokenizer
import xml.etree.ElementTree as ET


evaluation_file_path = "/export/users/liu/EvaluationDataset/"
result_text_path = "/export/users/liu/Result/text/"
dialog_xml_path = "/export/users/liu/datasets/Manga109Dialog/"
manga_name_list = []


def get_bert_embedding(text):
    # 加载BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")

    # 对文本进行编码
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取[CLS]位置的向量作为整个句子的表示
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding


def remove_tags(text):
    cleaned_text = re.sub(r"<Other>:", "", text)
    return cleaned_text


# 计算BLUE
def calculate_BLEU(result_text, evaluation_text):
    # 确保下载必要的资源
    nltk.download("punkt")
    # 使用 Janome 分词器对日语文本进行分词
    tokenizer = Tokenizer()

    # 分词并返回词语列表
    reference = [token.surface for token in tokenizer.tokenize(evaluation_text)]
    result = [token.surface for token in tokenizer.tokenize(result_text)]

    # 计算 BLEU 分数
    weights = (1, 0, 0, 0)  # 调整权重
    smooth_fn = SmoothingFunction().method2  # 使用平滑方法以减少低分数的影响
    bleu_score = sentence_bleu(
        [reference], result, smoothing_function=smooth_fn, weights=weights
    )

    return bleu_score


# 句子级别的语义相似度 BERT
def calculate_BERT(result_text, evaluation_text):
    # 获取两个文本的BERT向量
    result_text_embedding = get_bert_embedding(result_text)
    evaluation_text_embedding = get_bert_embedding(evaluation_text)

    # 计算余弦相似度
    similarity = cosine_similarity(result_text_embedding, evaluation_text_embedding)
    return similarity[0][0]


def extract_ids(xml_file, id_type):
    result_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for speaker in root.iter("speaker_to_text"):
        text_id = speaker.get(id_type)
        if text_id:
            result_list.append(text_id)

    return result_list


for filename in os.listdir(evaluation_file_path):
    manga_name_list.append(filename)


evaluation_text_fields = []
result_text_fields = []
text_id_list = []
sum_bleu_score = 0
sum_BERT_score = 0

for name in manga_name_list:
    try:

        name = os.path.splitext(name)[0]

        xml_file = dialog_xml_path + name + ".xml"
        text_id_list = extract_ids(xml_file, "text_id")

        # 评价文本
        print(evaluation_file_path + f"{name}")
        with open(
            evaluation_file_path + f"{name}.csv", "r", encoding="utf-8", errors="ignore"
        ) as file:
            reader = csv.reader(file)
            for row in reader:
                text_id = row[6]
                if row[7] == "text" and text_id in text_id_list:
                    evaluation_text_fields.append(row[8])
                    evaluation_text_fields.append("\n")

        # 结果文本
        print(result_text_path + f"{name}")
        with open(
            result_text_path + f"{name}" + "/transcript.txt", "r", encoding="utf-8"
        ) as file:
            lines = file.readlines()
            for line in lines:
                cleaned_text = remove_tags(line)
                result_text_fields.append(cleaned_text)

        # for text in result_text_fields:
        #     print(text)
    except Exception as e:
        print(f"发生错误: {e}")

    evaluation_text = " ".join(evaluation_text_fields)
    result_text = " ".join(result_text_fields)

    bleu_score = calculate_BLEU(result_text, evaluation_text)
    print(f"BLEU：{bleu_score:.2f}")

    sum_bleu_score = sum_bleu_score + bleu_score

    BERT_score = calculate_BERT(result_text, evaluation_text)
    print(f"BERT：{BERT_score:.2f}")

    sum_BERT_score = sum_BERT_score + BERT_score

    with open("evaluation_text.txt", "a", encoding="utf-8") as file:
        file.write(str(f"BLEU：：{bleu_score:.2f}") + "\n")
        file.write(str(f"BERT：{BERT_score:.2f}") + "\n")
        file.write("\n")

average_bleu_score = sum_bleu_score / len(manga_name_list)
average_BERT_score = sum_BERT_score / len(manga_name_list)

with open("evaluation.txt", "a", encoding="utf-8") as file:
    file.write(str(f"平均BLEU：{average_bleu_score:.2f}") + "\n")
    file.write(str(f"平均BERT：{average_BERT_score:.2f}") + "\n")

print(f"平均BLEU：{average_bleu_score :.2f}")
print(f"平均BERT：{average_BERT_score:.2f}")
