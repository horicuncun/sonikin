# sonikin
+ 串カツのソース二度づけ判定プログラム。
+ 事前に学習したモデルを使って、ソース二度づけしようとしているのを検知します。
+ 検知した場合は左上に「ソース二度付け禁止」と出ます。

# 必要なもの
+ Webカメラ
+ python3
+ numpy
+ cv2
+ keras
+ PIL

# 使い方
1. WebカメラをPCに接続
1. cd sonikin
1. python sonikin.py
1. Webカメラの前で串カツを食べる
1. ソース二度づけを検知した場合は左上に「ソース二度付け禁止」と出る

# うまくいかないときは
+ モデル(my_model.h5)を作り直すといいと思います
