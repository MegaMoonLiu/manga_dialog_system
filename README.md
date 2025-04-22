# 漫画画像からの対話システム
<p>現在の研究では、漫画に対話の内容やテキストボックス、キャラクターの識別に注目している。</p>
<p>これらの情報は対話システムの構築に有効ですが、マンガにはまだ見落(みお)とされている情報が多い。</p>
<p>漫画にあるキャラクターの表情と動き、性格によって、より立体的なキャラクターを作って、キャラクターにふさわしくて、感情込めている対話を作れる。</p>

### 現在やったこと
<p>漫画からキャラクターとコマの抽出</p>
<p>キャラクターの分類</p>
<img src="/result/page.png">


<a href="/result//transcript.txt">セリフの識別と抽出</a>

<p>抽出したセリフの評価</p>
evaluation
 平均BLEU：0.43
 平均BERT：0.87
 
<p>抽出したセリフとキャラクターのマッチ度</p>
1冊の評価
AkkeraKanjinchou evaluation
 目標キャラクター：['00001d7b', '00001d98', '00002367']
 抽出したキャラクターの数：864
 元のdataにある目標キャラクターの数：689
 正解率：0.65
 
### これからやること
<p>生成した対話の評価</p>
<p>Streamlitを利用してUIがある対話システムの構築</p>


# 関連研究
<a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov10-object-detection-on-custom-dataset.ipynb#scrollTo=SaKTSzSWnG7s">The Manga Whisperer: Automatically Generating Transcriptions for Comics </a>
<p>漫画画像からオブジェクトの検出と話者特定、発話の順番の特定を行う</p>


<a href="https://arxiv.org/abs/2401.10224">Ma`nga109</a>
<p>利用するアノテーション付きマンガ画像データセット。</p>

<a href="https://github.com/kha-white/manga-ocr">Manga ocr</a>
<p>セリフの識別と抽出</p>

<a href="https://github.com/manga109/public-annotations">Manga109Dialog</a>
<p>発話の順番付きデータセット</p>

<a href="https://arxiv.org/abs/2408.00298">Tails Tell Tales: Chapter-Wide Manga Transcriptions with Character Names</a>
<p>評価指標を参照した</p>
