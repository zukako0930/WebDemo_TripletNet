# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response,request
import cv2
import numpy as np
from flask import Flask, request

api = Flask(__name__)

@api.route("/predict", methods=['POST'])
def predict():
    print(request.files['file'].stream)
    stream = request.files['file'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    cv2.imwrite('image.png',img)
    return make_response(jsonify({"status":"success"}))


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=3001)