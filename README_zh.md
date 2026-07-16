# 漫画画像からの対話システム

<p align="center">
  <a href="README.md"><img alt="日本語" height="20" src="https://img.shields.io/badge/日本語-CDCFD4"></a>&nbsp;
  <a href="README_zh.md"><img alt="简体中文" height="20" src="https://img.shields.io/badge/简体中文-BCDCF7"></a>&nbsp;
</p>

<p>目前的研究主要关注于漫画中的对话内容、文本框以及角色识别。</p>
<p>虽然这些信息对构建对话系统非常有效，但漫画中仍有许多被忽略的信息。</p>
<p>通过结合漫画中角色的表情、动作和性格，可以塑造出更加立体、丰满的角色，从而生成更符合角色设定且富含情感的对话。</p>

### 研究目的
-  **提取各角色的台词：** 通过分析漫画中的文本框，准确提取每个角色的台词内容（包括文本、语气、遣词造句等），为对话系统提供基础的语料库数据。

-  **提取表情、背景和动作等信息：** 提取漫画的视觉信息、角色的表情、姿态、手势以及背景场景等。角色的表情和动作能够反映出他们在不同情境下的情感状态和行为特征。

-  **构建更加立体的角色形象：** 通过综合分析角色的台词、表情、动作和背景等信息，更立体地构建角色的性格、习惯和情感变化。这能让角色对各种情境做出更真实的反应。

- **实现基于角色的对话系统：** 利用提取的信息，开发能够模拟特定角色对话和行为的对话系统。该系统不仅能根据上下文生成自然语言，还能参考角色的表情、动作、背景等信息，从而构建出更具真实感的系统。

### Overview
<img src="/Example/Overview.png">
识别漫画中的角色，提取其动作与台词并导入到大语言模型（LLM）中，从而实现人机对话

### 目前已完成的工作
-  从漫画中提取角色和分镜
-  角色分类
    -  相同颜色的线条代表同一个角色
    -  蓝色框内为角色的名字
    -  红色框内为角色的台词
<img src="/Example/page.png">



-  台词的识别与提取
```
<流石あさはか>:ほっほっはっ何をかくそうこれは歯食人といっしょに作っ発見球でね
<Other>:まさに管香の思い出ってわけですね
<Other>:なんじゃ恋愛になんだってるわ
<流石あさはか>:中でも熱気球はダス気球などより比較的安全が９で安全性も高いだよ
<流石あさはか>:ダメっ！！
<Other>:あたしよっ！
<流石しとやか>:小学生ならふたりが限度ってとこだねジャンケンにしなさい！
<Other>:なーんだふたりかァ
<Other>:よりによってきるさんといっしょとはな．．．
<Other>:それはお母いさまだ！
<Other>:ねっお父さん
<Other>:気球の落ち心地ってどんな感じなの！？
<流石あさはか>:．．．
<流石あさはか>:ハカンわしがあんな危険な物に乗るかっ！
<Other>:迷ったのはお知らずな友人だ！
<Other>:もっとも理解不明の意識でも分ほど川に落ちて大ケかしちまったが．．．
```

### 对提取的台词进行评价
- 整体评价结果
  - 平均BLEU：0.43
  - 平均BERT：0.87
  
### 已提取台词与角色的匹配度
- AkkeraKanjinchou evaluation
  - 目标角色：['00001d7b', '00001d98', '00002367']
  - 提取出的角色数量：864
  - 原数据中目标角色数量：689
  - 正确率：0.65
 
### 利用Streamlit构建带UI的对话系统
<p>输入想要对话的角色图像和名字，即可开始对话</p>

<img src="https://github.com/MegaMoonLiu/manga_dialog_system/blob/main/Example/dialog_systeam_UI_1.png" width="50%" height="50%"><img src="https://github.com/MegaMoonLiu/manga_dialog_system/blob/main/Example/dialog_systeam_UI_2.png" width="50%" height="50%">

 
### 今后的工作
- 整体的角色匹配度
- 对生成的对话进行评价


# 相关研究
- <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s">The Manga Whisperer: Automatically Generating Transcriptions for Comics </a>
  - <p>从漫画图像中进行目标检测、对说话者进行识别，并确定说话顺序</p>


- <a href="https://arxiv.org/abs/2401.10224">Manga109</a>
  - <p>所使用的已标注的漫画图像数据集</p>

- <a href="https://github.com/kha-white/manga-ocr">Manga ocr</a>
  - <p>台词的识别与提取</p>

- <a href="https://github.com/manga109/public-annotations">Manga109Dialog</a>
  - <p>带有说话顺序的数据集</p>

- <a href="https://arxiv.org/abs/2408.00298">Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names</a>
  - <p>所参考的评价指标</p>
