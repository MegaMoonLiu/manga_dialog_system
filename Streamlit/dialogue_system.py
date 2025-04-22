import os
import openai

# 设置 API Key
openai.api_key = ""

dialogue_path = "Manga_Whisperer/transcript.txt"
image_folder_path = "Manga_Whisperer"

# 定义支持的图片格式
supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

dialogue = []
chapter_pages = [
    os.path.join(image_folder_path, filename)
    for filename in os.listdir(image_folder_path)
    if filename.lower().endswith(supported_extensions)
]

with open(dialogue_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        dialogue.append(line)

prompt = [
    f"用意されたコーパス{dialogue}"
    + f"と画像情報{chapter_pages}"
    + "に基づいて、キャラクターの話し方、表情、動作の癖を模倣し、対話を生成します。"
    + "また、ユーザーは対話するキャラクターを選択することもできます。"
]


# 对话系统函数
def chat_with_gpt(messages):
    """
    调用 OpenAI 的 GPT 模型，与用户进行多轮对话。

    参数:
    - messages: 包含对话上下文的消息列表

    返回:
    - GPT 的回复
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 或 "gpt-3.5-turbo"
            messages=messages,
            temperature=0.7,  # 控制生成的随机性，值越高越随机
            max_tokens=300,  # 每次回复的最大长度
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"发生错误: {e}"


# 初始化对话
if __name__ == "__main__":
    print("対話システムです！「exit」と入力すると、対話を終了します。")

    # 对话上下文存储
    conversation = [{"role": "system", "content": f"{prompt}"}]

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("ご利用ありがとうございました。")
            break

        # 添加用户消息到对话上下文
        conversation.append({"role": "user", "content": user_input})

        # 获取 GPT 回复
        reply = chat_with_gpt(conversation)
        print(f"GPT: {reply}")

        # 将 GPT 的回复添加到对话上下文
        conversation.append({"role": "assistant", "content": reply})
