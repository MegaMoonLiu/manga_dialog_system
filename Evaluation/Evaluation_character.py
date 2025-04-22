from janome.tokenizer import Tokenizer
import xml.etree.ElementTree as ET
import re
import numpy as np


# 标记了对话顺序的dialog 文件
dialog_file_path = "/datasets/Manga109Dialog/"
# 通过magiv2生成的文件
result_character_path = "/Result/character/"
# 原始的标注文件
evaluation_file_path = "/Manga109_released_2023_12_07/annotations.v2018.05.31/"


# 获得某一id_type的所有id
def extract_ids(xml_file, id_type):
    result_list = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for speaker in root.iter("speaker_to_text"):
        text_id = speaker.get(id_type)
        if text_id:
            result_list.append(text_id)

    return result_list


# 标记了对话顺序的dialog 文件
xml_file = dialog_file_path + "AkkeraKanjinchou.xml"
# dialog文件中的speaker_id
evaluation_dialog_character_list = extract_ids(xml_file, "speaker_id")

# 通过magiv2生成的结果
result_character_file = result_character_path + "AkkeraKanjinchou/character.txt"

# 所有角色的标签列表
all_characters_labels_list = []

# 过滤后的角色的标签列表
unique_characters_labels_list = []

with open(result_character_file, "r", encoding="utf-8") as r_c_f:
    content = r_c_f.read()
    for line in content.splitlines():
        # 替换 <None> 和 <unsure> 为 ""
        line = re.sub(r"<(None|unsure)>", "", line)

        # 使用正则表达式提取数字标签并保留顺序
        # 提取标签并将其转换为列表
        matches = re.findall(r"<([0-9a-fA-F]+)>", line)

        # 替换 <None> 和 <unsure> 为 ""，并保留标签顺序
        all_characters_labels_list.extend(matches if matches else ['""'])

# 去除重复标签并保持顺序
seen = set()
for label in all_characters_labels_list:
    if label not in seen:
        unique_characters_labels_list.append(label)
        seen.add(label)

unique_characters_labels_list = [
    label for label in unique_characters_labels_list if label != '""'
]

print("目標キャラクター：" + str(unique_characters_labels_list))


evaluation_file = evaluation_file_path + "AkkeraKanjinchou.xml"

# 从dialog文件中获取所有顺序标注的对话记录
dialog_file_tree = ET.parse(xml_file)
dialog_file_root = dialog_file_tree.getroot()

speaker_to_text_mapping = {}

# 获取所有speaker_id和与之对应的text_id
for speaker_to_text in dialog_file_root.findall(".//speaker_to_text"):
    speaker_id = speaker_to_text.get("speaker_id")
    text_id = speaker_to_text.get("text_id")
    speaker_to_text_mapping[speaker_id] = text_id


#  获取与speaker_to_text_mapping中对齐的character list
evaluation_file_tree = ET.parse(evaluation_file)
evaluation_file_root = evaluation_file_tree.getroot()
evaluation_character_list = []

for speaker_id, text_id in speaker_to_text_mapping.items():
    for body in evaluation_file_root.findall(".//body"):
        if body.get("character") and body.get("id") == speaker_id:
            evaluation_character_list.append(body.get("character"))

# print(all_characters_labels_list)
# print(evaluation_character_list)


# 计算目标元素在列表中的所有位置
def find_positions(lst, target):
    return [i for i, x in enumerate(lst) if x == target]


# 计算顺序匹配度
# 通过动态规划算法计算两个子序列的最长公共子序列（LCS）
# 将公共子序列长度与两个子序列的最大长度进行比较，得到顺序匹配度
def calculate_matching_degree(list1, list2, target_elements):
    # 提取两个列表中仅包含目标元素的子序列
    sub_seq1 = [x for x in list1 if x in target_elements]
    sub_seq2 = [x for x in list2 if x in target_elements]

    # 计算最长公共子序列长度
    def lcs_length(a, b):
        m = len(a)
        n = len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs_len = lcs_length(sub_seq1, sub_seq2)
    max_len = max(len(sub_seq1), len(sub_seq2))

    # 避免除以0的情况
    if max_len == 0:
        return 0.0

    matching_degree = lcs_len / max_len
    return matching_degree


matching_degree = calculate_matching_degree(
    all_characters_labels_list, evaluation_character_list, unique_characters_labels_list
)


# 输出结果
print(f"抽出したキャラクターの数：{len(all_characters_labels_list)}")
print(f"元のdataにある目標キャラクターの数：{len(evaluation_character_list)}")
print(f"マッチ度：{matching_degree:.2f}")
