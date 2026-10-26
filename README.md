# multi-layer-cnn

- Main: branch of original directory in nero 2019

- wsl: branch of new directory on local wsl environment after 2020 for SUIKO

## 概要

CMIP 17モデルで学習したCNNにより、5月から10月の月降水量（APHRODITE）をリードタイム1〜3か月で予測し、
SODA/GODASによる転移学習、ECMWF・GFDLとの比較、ブートストラップによる信頼区間推定、作図を行う。

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `cnn_common.py` | `cnn_cmip.py` と `cnn_trans.py` の共通部分（データ切り出し `cmipMon`/`cmipMon53`、`shuffle`、CNNモデル定義、対象月×リードタイムのループ、スコア表示） |
| `cnn_cmip.py` | 実行スクリプト。CMIPのみでCNNを学習（アンサンブル5）し、相関とRMSEを保存 |
| `cnn_trans.py` | 実行スクリプト。CMIPで事前学習後、SODAで転移学習しGODASで検証（アンサンブル20） |
| `bootstrap.py` | 実行スクリプト。GFDL（およびコメントアウトされたcnn_trans、APHRO）のブートストラップ信頼区間 |
| `plot.py` | 実行スクリプト。相関スキルの棒グラフなどを作図 |
| `training_<月>/` | `cnn_trans.py` が書き出すKerasのチェックポイント（カレントディレクトリ基準の相対パス） |

## 実行方法

すべてのスクリプトは、カレントディレクトリの一つ上にある `output/` を入力元とする（`os.getcwd()` 基準）。
このリポジトリ直下で実行すること。

```bash
python cnn_cmip.py    # -> /home/Hasegawa/Dthesis/water/output/cnn_cmip/leadtime_{1,2,3}month/cmip_<月>.npy
python cnn_trans.py   # -> ./training_<月>/cp.ckpt（予測値の保存は元コードどおりコメントアウト）
python bootstrap.py   # -> ../output/GFDL/leadtime_*/bootstrap_<月>.npy
python plot.py        # -> ../pic/cnn_trans/multibar_leadtime_*.png
```

入力ファイル（`../output/` 内）
`inp_cmip17.npy`, `pr_cmip17.npy`, `aphro_month_1951-2015.npy`, `inp_godas.npy`, `soda_train_for_pr.npy`,
`ECMWF/`, `GFDL/`, `cnn_trans/`, `cnn_cmip/` 以下の各npy。

5月・リードタイム3か月（`m=5, gap=6`）だけは `m-gap` が負になるため `cmipMon53` / `transDset53` を使う。

## 必要パッケージ

numpy, pandas, matplotlib, seaborn, tensorflow (2.x)
