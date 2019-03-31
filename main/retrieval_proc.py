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

# # [[id, 画像, 埋め込みベクトル],[id, 画像, 埋め込みベクトル], ...]の検索対象データを予め読み込んでおく
# # pickleにして保存しておいて読み込む
# import pickle 
# with open('gallery.pickle', 'rb') as g:
#   gallery = pickle.load(g)
#   print(gallery[0])

# 以下の関数はinfer.pyからの呼び出しで実行される

# sample data
def query():
    PATH = './image.png'
    return np.array(Image.open(PATH).resize((128,128)).convert('RGB'))/255.

# idsをもとにshop画像をベクトル化する/idsは数字のみを想定
# 全てのidの場合はids = os.listdir(PATH)
def get_shopimgs(ids,BASE_PATH):
    shop_img_list=[]
    for id_ in ids:
        file_paths = sorted([BASE_PATH+id_+'/'+x for x in os.listdir(BASE_PATH+id_)])
        shop_paths = sorted([x for x in file_paths if 'shop' in x ])
        for shop_path in shop_paths:
            img = np.array(Image.open(shop_path).resize((128,128)).convert('RGB'))/255.
            # conにする？
            emb = shop_embNet.predict(np.expand_dims(img.astype(np.float32),axis=0))[0]
            shop_img_list.append([id_[:],shop_path,emb])
    return shop_img_list

# 近傍探索用
def distance(emb1,emb2):
    return np.sum(np.square(emb1-emb2))

# ユーザが入力した画像データの埋め込み
def con_embedding(img):
    # plt.imshow(img)
    print(img.shape)
    img = img/255. # curlで送られてきた画像を0~1に収める。
    # plt.imshow(img) # 勝手にimshowが0~255に正規化する
    # plt.show()
    #shopでもやる
    return shop_embNet.predict(np.expand_dims(img,axis=0))[0] # need [0] because of expanding dimension -> [[]]

# ユーザの入力画像に対してランキングを計算
def calc_ranking(query_img_emb,gallery):
    print(type(gallery))
    N = 50
    # topN_id_list=[] # [[query1->上位20個のid],...,[queryK->上位20個のid]]
    similarities = np.zeros(len(gallery))
    for i,g in tqdm(enumerate(gallery)):
        similarities[i] = distance(query_img_emb,g[2]) # retrieve from all the images
    idxs = similarities.argsort() # ascending order/ argsort() returns indexes
    print(idxs[:10])
    print(type(np.array(gallery)))
    np_gallery = np.array(gallery)
    ranking = list(np_gallery[idxs[:N]]) #類似度の高い順にランキングしたインデックスでgalleryを切り出す
    # topN_id_list.append(gallery[idxs[:N]]) # append ids
    rank_list=[]
    for r in ranking:
        my_dic = {}
        my_dic['image_path'] = './public'+ r[1][1:]
        my_dic['image_id'] = r[0]
        rank_list.append(my_dic)
    return rank_list
