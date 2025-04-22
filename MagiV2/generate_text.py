import PIL.Image
import os
import re
from manga_ocr import MangaOcr

mocr = MangaOcr()
image_folder_path = "/Manga_Whisperer/crops_output/"
text_path = "/Manga_Whisperer/dialogue/"


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
