# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response,request
import cv2
from PIL import Image
import numpy as np
from flask import Flask, request
import retrieval_proc
from io import BytesIO

api = Flask(__name__)

# predictのエンドポイントのアクセスで呼ばれる関数
# curl -F "file=@a.png"  http://0.0.0.0:3001/predict
@api.route("/predict", methods=['POST'])
def predict():
    print(request.files['file'].stream)
    stream = request.files['file'].stream
    arr = (Image.open(BytesIO(stream.read())))
    print(arr.mode)
    img = np.asarray(arr.resize((128,128)).convert("RGB"))
    # 入力画像をベクトル化
    query_img = retrieval_proc.con_embedding(img)
    # ランキングを算出して結果を表示
    print(query_img)

    return make_response(jsonify({"status":"success"}))


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)