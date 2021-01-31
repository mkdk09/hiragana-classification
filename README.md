# ひらがな認識
Webページ上のCanvasに文字を書いて，ひらがなを認識するWebアプリ  
Kerasで学習したモデルをTensorFlow.jsで読み込めるようにモデルを変換している．

## デモ(DEMO)
https://mkdk09.github.io/hiragana-classification/

## 機能(Features)
* Canvas上に黒の文字を書くことができます．
* Canvasに書かれた文字を読み込み，学習済みモデルに渡してひらがなを認識
* 認識した文字のところをグレーにして精度を表示する

## 使い方(Usage)

## 環境(Requirement)

## インストール(Installation)

## 注意事項(Note)
レスポンシブではないのでスマホ等でページを開くとレイアウトが崩れます．

## 文責(Author)
* mkdk09
* mkdk099@gmail.com

## ライセンス
This code is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

## 参考文献(References)
* [TensorFlow.jsでMNIST学習済モデルを読み込みブラウザで手書き文字認識をする - Qiita](https://qiita.com/yukagil/items/ca84c4bfcb47ac53af99)   
* [TensorFlow.jsを使ってKerasで作成したモデルを利用してみる | みんな栄養に頼りすぎてる](https://www.y-shinno.com/tensor-flow-js-mnist/)  

### 使用データセット
* [ndl-lab/hiragana_mojigazo: 文字画像データセット(平仮名73文字版)](https://github.com/ndl-lab/hiragana_mojigazo/)
