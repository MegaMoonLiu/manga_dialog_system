from PIL import Image
import PIL.Image
import numpy as np
from transformers import AutoModel
import torch
import os
from manga_ocr import MangaOcr
import json
import re
import shutil

dialogue_path = "dialogue/"
crops_output_path = "crops_output/"
character_txt_path = "character.txt"
result_json_path = "results.json"
transcript_path = "transcript.txt"


def delete_all_contents(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print(file_path)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # 删除文件和符号链接
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # 递归删除文件夹


def delete_file(file_path):
    if os.path.exists(file_path):  # 确保文件存在
        os.remove(file_path)
        print(f"文件 {file_path} 已删除")
    else:
        print(f"文件 {file_path} 不存在")


# 示例调用
delete_all_contents(dialogue_path)
delete_all_contents(crops_output_path)
delete_file(character_txt_path)
delete_file(result_json_path)
delete_file(transcript_path)
# 清空预存结果


# 加载模型
model = (
    AutoModel.from_pretrained(
        "/export/users/liu/MagiV2",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    .cuda()
    .eval()
)


def read_image(path_to_image):
    with open(path_to_image, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image


# 设置图片文件夹路径
image_folder_path = "/export/users/liu/datasets/BokuHaSitatakaKun"

# 定义支持的图片格式
supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# 获取所有图片文件的路径
chapter_pages = [
    os.path.join(image_folder_path, filename)
    for filename in os.listdir(image_folder_path)
    if filename.lower().endswith(supported_extensions)
]

character_bank = {
    "images": [
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b31a/0000b3b7.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b31e/0000b3a1.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b320/0000b4bb.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b32a/0000b3c7.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b333/0000b6a5.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b375/0000b3a0.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b3cf/0000b46f.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b3f5/0000b3f4.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b4ab/0000b4d9.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000b598/0000b5a5.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000c2a7/0000c2a6.jpg",
        "/export/users/liu/manga-face-clustering-1.0.1/dataset/images/0000ce12/0000ce5c.jpg",
    ],
    "names": [
        "斉藤つとむ",
        "本郷勇一",
        "大貫アキラ",
        "流石したたか",
        "本田原海",
        "流石あざやか",
        "流石おろそか",
        "流石おちゃらか",
        "流石しとやか",
        "水越けいこ",
        "流石あさはか",
        "本田原空",
    ],
}

# 读取并调整图像大小
chapter_pages = [read_image(x) for x in chapter_pages]
character_bank["images"] = [read_image(x) for x in character_bank["images"]]

# 使用模型进行预测
with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(
        chapter_pages, character_bank, use_tqdm=True, do_ocr=True
    )

# 生成台词
mocr = MangaOcr()
image_folder_path = "crops_output/"
text_path = "dialogue/"

files = sorted(
    [
        f
        for f in os.listdir(image_folder_path)
        if f.startswith("image_") and f.endswith(".png")
    ]
)


def extract_number(file_name):
    parts = file_name.split("_")  # 按下划线分割
    x = int(parts[1])  # 提取第一个数字 X
    y = int(parts[3].split(".")[0])  # 提取第二个数字 Y
    return (x, y)  # 返回元组用于排序


sorted_files = sorted(files, key=extract_number)

for filename in sorted_files:
    # 从文件名中提取数字部分
    page = filename.split("_")[1]  # 获取文件名
    dialogue = filename.split("_")[3].split(".")[0]  # 提取数字部分
    image_file_path = os.path.join(image_folder_path, filename)
    img = PIL.Image.open(image_file_path)
    text = mocr(img)
    print(page, dialogue)
    with open(text_path + f"transcript_{page}.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(image_file_path)

# 生成角色
character = []
for i, (image, page_result) in enumerate(zip(chapter_pages, per_page_results)):
    model.visualise_single_image_prediction(image, page_result, f"page_{i}.png")
    speaker_name = {
        text_idx: page_result["character_names"][char_idx]
        for text_idx, char_idx in page_result["text_character_associations"]
    }
    for j in range(len(page_result["ocr"])):
        if not page_result["is_essential_text"][j]:
            continue
        name = speaker_name.get(j, "unsure")
        character.append(f"<{name}>: " + "\n")

# # 匹配台词
json_path = ""
page = 0

# 读取 JSON 文件
with open(json_path + "results.json", "r", encoding="utf-8") as file:
    raw_content = file.read()

# 修复 JSON：将多个独立的 JSON 块组合为一个列表
fixed_json = raw_content.replace("][", ",")

try:
    results = json.loads(fixed_json)
    print("JSON 解析成功")
except json.JSONDecodeError as e:
    print(f"JSON 解析失败: {e}")

dialogue = []
for page in range(len(results)):
    try:
        with open(text_path + f"transcript_{page}.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            j = 0
            for value in results[page][f"{page}"]:
                if value:
                    dialogue.append(lines[j])
                j += 1
    except Exception as e:
        print(f"发生错误: {e}")
print(f"dialogue: {len(dialogue)}")

# 保存转录文本
with open("character.txt", "w", encoding="utf-8", errors="ignore") as fh:
    for i in character:
        fh.write(i)

with open("character.txt", "r", encoding="utf-8") as c_f:
    character = []
    lines = c_f.readlines()
    for line in lines:
        line = line.strip()
        character.append(line)
    print(f"character: {len(character)}")

with open("transcript.txt", "a", encoding="utf-8") as r_f:
    for i in range(len(character)):
        r_f.write(character[i] + dialogue[i])
