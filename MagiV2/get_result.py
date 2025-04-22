import json
import os
import re

json_path = "/Manga_Whisperer/"
text_path = "/Manga_Whisperer/dialogue/"

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


# print(results[0])
# for value in results[0]["0"]:
#     print(value)


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

print(dialogue)
print(len(dialogue))

character = []

with open(json_path + "transcript.txt", "r", encoding="utf-8") as c_f:
    lines = c_f.readlines()
    for line in lines:
        line = line.strip()
        character.append(line)

print(character)
print(len(character))

with open("result.txt", "w", encoding="utf-8") as r_f:
    for i in range(len(character)):
        r_f.write(character[i] + dialogue[i])
