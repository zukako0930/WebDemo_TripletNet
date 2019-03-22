#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct

# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response,request
import flask
import base64
from io import BytesIO

api = Flask(__name__)

import os
BASE_PATH = './T_Shirt_all/'

# modelの配置
# from keras.models import load_model
# model_epoch = 20
# con_embNet = load_model('./model/T_Shirt/krasser/con_emb_e{}.h5'.format(model_epoch))
# shop_embNet = load_model('./model/T_Shirt/krasser/shop_emb_e{}.h5'.format(model_epoch))

@api.route('/retrieval',methods=['POST'])
def retrieval():
    print(flask.request.data)
    enc_data = flask.request.data # 渡されたバイナリ画像データを読み込み
     # デコードする 
    # print(base64.b64decode(enc_data))
    # dec_data = base64.b64decode(enc_data) # デコードする 
    # image_decoded = Image.open(BytesIO(dec_data))
    # jpg=np.frombuffer(dec_data,dtype=np.uint8)
    # image = Image.open(jpg)
    # img = Image.open(StringIO(buffer))
    # image_decoded.save('image.png')
    return make_response(jsonify({'result':'success'}))


if __name__ == '__main__':
    api.run(debug=True)