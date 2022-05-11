# Feature Extractors

* 良い感じの表現(特にcosine類似度観点)を獲得した学習済みモデルの出力をそのままLightGBMとかにぶち込みたい
* 実戦では最初に全データに対して表現を出力したほうが便利そう
* DataLoaderとか実装はデータによるのでその場で実装で良さそう

## Image
[CLIP](https://github.com/openai/CLIP)

* 使い方はREADMEを読めばすぐにわかる
* 抽出特徴量をロジスティック回帰の学習に使う例: https://github.com/openai/CLIP#linear-probe-evaluation
* 使えるモデルはこの辺: https://github.com/openai/CLIP/blob/main/clip/clip.py
* 対応するImageとTextのcosine類似度が近くなるように学習させており、特にImageに限る必要はなく、Textの表現も使えないことない気がする

## Text
[SimCSE](https://github.com/princeton-nlp/SimCSE)

* 使い方はREADMEを読めばすぐにわかる(`str`を`encode`に食わせるだけ)
* 使えるモデルはREADMEにある
