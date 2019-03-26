# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response,request,render_template
import cv2
from PIL import Image
import numpy as np
from flask import Flask, request
from io import BytesIO

import matplotlib.pyplot as plt

# 自作モジュールの読み込み
import retrieval_proc

# apiの宣言
api = Flask(__name__, template_folder='.',static_folder='./public')

# [[id, 画像, 埋め込みベクトル],[id, 画像, 埋め込みベクトル], ...]の検索対象データを予め読み込んでおく
# pickleにして保存しておいて読み込む
import pickle 
with open('gallery.pickle', 'rb') as g:
  gallery = pickle.load(g)
  print(gallery[0])

@api.route('/')
def index():
  return render_template('./public/index.html')

# predictのエンドポイントのアクセスで呼ばれる関数
# curl -F "file=@a.png"  http://0.0.0.0:3001/predict
@api.route("/predict", methods=['POST'])
def predict():
    print(request.files['file'].stream)
    # print(request.form['file'].stream)
    stream = request.files['file'].stream
    arr = (Image.open(BytesIO(stream.read())))
    print(arr.mode)
    img = np.asarray(arr.resize((128,128)).convert("RGB"))
    # 入力画像をベクトル化
    query_img_emb = retrieval_proc.con_embedding(img)
    # print(query_img_emb)

    # ランキングを算出して結果を表示
    rank_list = retrieval_proc.calc_ranking(query_img_emb,gallery)
    # rank_listに変数を代入してresult.htmlにパラメータとして渡す
    return render_template('./public/result.html',rank_list=rank_list)
    # return make_response(jsonify({"status":"success"}))

if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)