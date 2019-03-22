# ライブラリのinstall
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# modelの読み込み
from keras.models import load_model
model_epoch = 25 # 何エポックのモデルを利用するか
con_embNet = load_model('./model/T_Shirt/krasser/con_emb_e{}.h5'.format(model_epoch))
shop_embNet = load_model('./model/T_Shirt/krasser/shop_emb_e{}.h5'.format(model_epoch))

# sample data
def query():
    PATH = './image.png'
    return np.array(Image.open(PATH).resize((128,128)).convert('RGB'))/255.
# [[id, 画像],[id, 画像], ...]の検索対象データを用意


# 近傍探索用
def distance(emb1,emb2):
    return np.sum(np.square(emb1-emb2))

# 画像データの埋め込み
def con_embedding(img):
    # plt.imshow(img)
    print(img.shape)
    img = img/255. # curlで送られてきた画像を0~1に収める。
    plt.imshow(img) # 勝手にimshowが0~255に正規化する
    plt.show()
    return shop_embNet.predict(np.expand_dims(img,axis=0))[0] # need [0] because of expanding dimension -> [[]]

# print(con_embedding(img))