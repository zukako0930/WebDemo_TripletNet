# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response,request
import cv2
from PIL import Image
import numpy as np
from flask import Flask, request
import retrieval_proc
from io import BytesIO

api = Flask(__name__)

@api.route("/predict", methods=['POST'])
def predict():
    print(request.files['file'].stream)
    stream = request.files['file'].stream
    arr = (Image.open(BytesIO(stream.read())))
    print(arr.mode)
    img = np.asarray(arr.resize((128,128)).convert("RGB"))
    # img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    # img = cv2.imdecode(img_array, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # img = cv2.bitwise_not(img)
    # cv2.imwrite('image.png',img)
    result = retrieval_proc.con_embedding(img)
    print(result)

    return make_response(jsonify({"status":"success"}))


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)