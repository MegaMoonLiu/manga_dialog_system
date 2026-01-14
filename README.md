# 漫画画像からの対話システム
<p>現在の研究では、漫画に対話の内容やテキストボックス、キャラクターの識別に注目している。</p>
<p>これらの情報は対話システムの構築に有効ですが、マンガにはまだ見落とされている情報が多い。</p>
<p>漫画にあるキャラクターの表情と動き、性格によって、より立体的なキャラクターを作って、キャラクターにふさわしくて、感情込めている対話を作れる。</p>

### 目的
-  **各キャラクターのセリフを抽出：** コミック内のテキストボックスを分析することで、各キャラクターのセリフ内容(文章、口調、言葉の選択など)を正確に抽出することで、対話システムに基本的なコーパスデータを提供する。

-  **表情、背景、行動などの情報を抽出：** マンガの視覚情報、キャラクターの表情、姿勢、手のジェスチャー、背景のシーンなどを抽出して、表情や行動は、キャラクターの感情状態や行動特性をさまざまな状況で反映することができます。

-  **より立体的なキャラクターを構築：** キャラクターのセリフ、表情、動作、背景などの情報を総合的に分析することでキャラクターの性格、習慣、感情の変化などをより立体的なキャラクターの構築ができる。これにより、キャラクターが各状況に対してより現実的な反応ができる。

- **キャラクターに基づく対話システムの実現：** 抽出された情報を使用し、キャラクターに基づく対話や行動をシミュレートできる対話システムを開発する。このシステムは、対話文に基づいて自然な文章を生成できるだけでなく、登場人物の表現、行動、背景などの情報を参照し、より現実的なシステムを構築する。

### Overview
<img src="/Example/Overview.png">
漫画内のキャラクターを識別して、動きとセリフを抽出(ちゅうしゅつ)して、LLMに導入(どうにゅう)して、人と会話できる

### 現在やったこと
-  漫画からキャラクターとコマの抽出
-  キャラクターの分類
<img src="/Example/page.png">


-  セリフの識別と抽出
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

### 抽出したセリフの評価
- 全体の評価結果
  - 平均BLEU：0.43
  - 平均BERT：0.87
  
### 抽出したセリフとキャラクターのマッチ度
- AkkeraKanjinchou evaluation
  - 目標キャラクター：['00001d7b', '00001d98', '00002367']
  - 抽出したキャラクターの数：864
  - 元のdataにある目標キャラクターの数：689
  - 正解率：0.65
 
### Streamlitを利用してUIがある対話システムの構築
<p>対話したいキャラクターの画像と名前を入力すると、会話が始まる</p>

<img src="https://github.com/MegaMoonLiu/manga_dialog_system/blob/main/Example/dialog_systeam_UI_1.png" width="50%" height="50%"><img src="https://github.com/MegaMoonLiu/manga_dialog_system/blob/main/Example/dialog_systeam_UI_2.png" width="50%" height="50%">

 
### これからやること
- 全体のキャラクターのマッチ度
- 生成した対話の評価


# 関連研究
<a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s">The Manga Whisperer: Automatically Generating Transcriptions for Comics </a>
<p>漫画画像からオブジェクトの検出と話者特定、発話の順番の特定を行う</p>


<a href="https://arxiv.org/abs/2401.10224">Manga109</a>
<p>利用するアノテーション付きマンガ画像データセット。</p>

<a href="https://github.com/kha-white/manga-ocr">Manga ocr</a>
<p>セリフの識別と抽出</p>

<a href="https://github.com/manga109/public-annotations">Manga109Dialog</a>
<p>発話の順番付きデータセット</p>

<a href="https://arxiv.org/abs/2408.00298">Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names</a>
<p>評価指標を参照した</p>
