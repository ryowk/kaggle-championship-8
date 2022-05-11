# kaggle-championship

## 開発方法

ディレクトリ構造
```
.
├── input
│   ├── test.csv
│   └── train.csv
└── wktk_0_model
    ├── main.py
    ├── output
    │   └── hoge
    │       ├── test.csv
    │       ├── train_0.csv
    │       ├── train_1.csv
    │       └── train_2.csv
    └── run.sh
```

* モデルは`*_model`ディレクトリ内に作る
* `*_model`内で`./run.sh`を実行すると`output/<任意の識別子>`内に予測結果を出力する
* `test.csv`: テストデータに対する予測。idと予測値を想定。
* `train_*.csv`: 各foldに対する予測。idと予測値を想定。
* モデルは1人1つに限定しない
* ディレクトリで分けているのでmainにダイレクトpushでよい

## outputの共有方法
gitに上げられないサイズの場合はS3で共有する

### ローカル -> S3 (modelごとに各自で)
```sh
aws s3 sync sample_model/output/ s3://pigimaru-kaggle-days-2021/championship-1/sample_model/
# 消すときはコンソールから手動か、ローカルで消してから以下
# aws s3 sync sample_model/output/ s3://pigimaru-kaggle-days-2021/championship-1/sample_model/ --delete
```

### S3 -> ローカル (ensembleするマンだけやればいい)
```sh
./download_from_s3.sh
# ローカルに余分なのあるとそれを消してくれないので注意
```

## アンサンブルの手順
```console
python ensembler.py
```
`importance.json`と`submission.csv`を出力する
