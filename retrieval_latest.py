#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Flaskに必要なモジュールの読み込み
from flask import Flask, jsonify, abort, make_response

import os
BASE_PATH = './T_Shirt_all/'

from keras.models import load_model
model_epoch = 20
con_embNet = load_model('./model/T_Shirt/krasser/con_emb_e{}.h5'.format(model_epoch))
shop_embNet = load_model('./model/T_Shirt/krasser/shop_emb_e{}.h5'.format(model_epoch))

# 画像一枚読み込み
# @api.roiute("/predict", methods=['POST'])
# def predict():
#     result = {
#         "greeting":'hello flask'
#     }
#     return make_response(jsonify(result))

# ### 既に分割したtestデータを呼んでくる
f = open("./test_ids.txt","rb")
test_ids = pickle.load(f)

# 自作moduleのimport
import datagen
test_pairs = datagen.get_test_pairs(test_ids,BASE_PATH,seed_num=0)
print("test_pairsの中身 : [product_id,['con_path','shop_path']] \ntest_pairs[0] : %s" %test_pairs[0])

import pickle
f = open('test_pairs.txt', 'wb')
pickle.dump(test_pairs, f)

# ### 検索画像プール(gallery)作成
# - idをintに直す必要がある?gallery=[id(string),[np.ndarray]]にしたらどう？
# - とりあえずtestだけのgalleryでやってみよう
aaa =[['id_0000000',np.zeros(10)] ,['id_0000001',np.ones(10)]]
gallery = []
for tp in test_pairs:
    gallery.append([tp[0],tp[1][1]])
gallery[0]

vec_length=100
# 予め入れ物を用意or後でappend
emb_vecs = np.zeros((len(gallery),vec_length))
ans_ids = np.zeros(len(gallery))
# gallery=[id,img_path]

for i,g in enumerate(tqdm(gallery)):
    img =np.array(Image.open(g[1]).resize((128,128)).convert('RGB'))/255.
    ans_ids[i] = g[0]
    emb_vecs[i] = shop_embNet.predict(np.expand_dims(img,axis=0))[0] # need [0] because of expanding dimension -> [[]]

# ユークリッド距離を計算する関数distance
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

query_ids = np.zeros(len(test_pairs))
for i,tp in enumerate(test_pairs):
    query_ids[i] = tp[0]

N = 50
topN_id_list=[] # [[query1->上位20個のid],...,[queryK->上位20個のid]]
for tp in tqdm(test_pairs):
    img = np.array(Image.open(tp[1][0]).resize((128,128)).convert('RGB'))/255.
    test = shop_embNet.predict(np.expand_dims(img.astype(np.float32),axis=0))[0]
#     test = con_embNet.predict(np.expand_dims(img.astype(np.float32),axis=0))[0]

    # calc order
    similarities = np.zeros(len(emb_vecs))
    for i,emb_vec in enumerate(emb_vecs):
        similarities[i] = distance(test,emb_vec) # retrieve from all the images
    idxs = similarities.argsort() # ascending order/ argsort() returns indexes
    topN_id_list.append(ans_ids[idxs[:N]]) # append ids


topN_id_list[0]

lim = 50
cnt = 0
top20_ids = []
topN_ids = []
topN_idxs = []
for i,(ans_id,topN_id) in enumerate(zip(ans_ids,topN_id_list)):
    if ans_id in topN_id[:lim]:
        cnt+=1
        topN_ids.append(ans_id)
        topN_idxs.append(i)
    if ans_id in topN_id[:20]:
#         top20_ids.append(i)
        top20_ids.append(ans_id)
#         print(ans_id)
print("{0}-top:{1}".format(lim,cnt/len(test_ids)))
# print(top20_ids)

def calc_topk_vals(k_values,ans_ids,topN_id_list,length):
    topk_vals = []
    for k in k_values:
        cnt = 0
        for i,(ans_id,topN_id) in enumerate(zip(ans_ids,topN_id_list)):
            if ans_id in topN_id[:k]:
                cnt+=1
        topk_vals.append(cnt/length)
    return topk_vals

mynet=calc_topk_vals(x,ans_ids,topN_id_list,len(test_ids))

mynet

x=[1,5,10,20,30,40,50]
fashionNet=[0.07,0.12,0.15,0.188,0.21,0.22,0.225]
# mynet=[0.011,0.0325,0.055,0.0885,0.11958,0.147,0.166]
mynet=calc_topk_vals(x,ans_ids,topN_id_list,len(test_ids))
chancelv=np.array(x)/len(test_pairs)

plt.plot(x,fashionNet,marker='o',label='FashionNet(0.188)')
plt.plot(x,mynet,c='r',marker='o',label='mine({})'.format(round(mynet[3],3)))
plt.plot(x,chancelv,linestyle='dashed',label='chance level(0.00984)')
plt.legend(loc='upper left')#,bbox_to_anchor=(1, 0.2))
plt.grid('True')
plt.title("top-k accuracy")
plt.xlabel("k-value")
plt.ylabel("top-k acc")

plt.savefig('./result_img/result_epoch{}.png'.format(model_epoch))


# 予測するときにresizeしているだけで元の画像を(128,128)にしているわけではないので比率を保って表示できる。

# queryと結果のidが同じときに枠線の色を変えたい
goodidx = np.array(topN_idxs[10:30])
print(goodidx)
# リストのスライスのため
from operator import itemgetter
results = itemgetter(*goodidx)(test_pairs)

def show_correct(img):
    img = np.array(img)
    img = np.array([np.pad(img[:,:,0], (8,8), 'constant', constant_values=(255,255)),np.pad(img[:,:,1], (8,8), 'constant',constant_values=(0,0)),np.pad(img[:,:,2], (8,8), 'constant',constant_values=(0,0))])
    plt.imshow(img.transpose(1,2,0))

imheight,imwidth=128,128
num = 5
for tp in results:
    # show query
    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,6,1)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    raw_img = Image.open(tp[1][0])
    img =np.array(raw_img.resize((128,128)).convert('RGB'))/255.
    test = shop_embNet.predict(np.expand_dims(img.astype(np.float32),axis=0))[0]
    plt.title('query',fontsize=18)
    show_correct(raw_img)

    # calc order
    similarities = np.zeros(len(emb_vecs))
    for i,emb_vec in enumerate(emb_vecs):
        similarities[i] = distance(test,emb_vec)
    idxs = similarities.argsort() #ascending order b.c. smaller is better/ get the index array in ascending order
    
    # show nearest
    plt.subplot(1,6,2)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    nearest = Image.open(gallery[idxs[0]][1])
    plt.title('1st',fontsize=18)
    if gallery[idxs[0]][0] == tp[0]:
        show_correct(nearest)
    else:
        plt.imshow(nearest)
    
    # show 2nd
    plt.subplot(1,6,3)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    second =  Image.open(gallery[idxs[1]][1])

    if gallery[idxs[1]][0] == tp[0]:
        show_correct(second)
    else:
        plt.imshow(second)
        
    plt.title('2nd',fontsize=18)
    
    # show 3rd
    plt.subplot(1,6,4)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    third =  Image.open(gallery[idxs[2]][1])
    plt.title('3rd',fontsize=18)
    if gallery[idxs[2]][0] == tp[0]:
        show_correct(third)
    else:
        plt.imshow(third)
    
    # show 4th
    plt.subplot(1,6,5)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    fourth =  Image.open(gallery[idxs[3]][1])
    plt.title('4th',fontsize=18)
    if gallery[idxs[3]][0] == tp[0]:
        show_correct(fourth)
    else:
        plt.imshow(fourth)
    
    
        # show 5th
    plt.subplot(1,6,6)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='white')
    fifth =  Image.open(gallery[idxs[4]][1])
    plt.title('5th',fontsize=18)
    if gallery[idxs[4]][0] == tp[0]:
        show_correct(fifth)
    else:
        plt.imshow(fifth)
    
    
    plt.savefig('./result_img/{}_result_id{}.png'.format(model_epoch,tp[0]),bbox_inches="tight")

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


imgs = os.listdir('./result_img/')
im1= Image.open('./result_img/'+imgs[0])
im2= Image.open('./result_img/'+imgs[1])
dst1 = get_concat_v(im1,im2)
im1= Image.open('./result_img/'+imgs[2])
im2= Image.open('./result_img/'+imgs[3])
dst2 = get_concat_v(im1,im2)
im1= Image.open('./result_img/'+imgs[4])
im2= Image.open('./result_img/'+imgs[5])
dst3 = get_concat_v(im1,im2)
im1= Image.open('./result_img/'+imgs[6])
im2= Image.open('./result_img/'+imgs[7])
dst4 = get_concat_v(im1,im2)
im1= Image.open('./result_img/'+imgs[8])
im2= Image.open('./result_img/'+imgs[9])
dst5 = get_concat_v(im1,im2)

dst6 = get_concat_v(dst1,dst2)
dst7 = get_concat_v(dst3,dst4)
dst8 = get_concat_v(dst6,dst7)
get_concat_v(dst8,dst5).save('./result_img/concat.png')

img = np.array(Image.open(test_pairs[1][1][0]))
print(img.shape)
print(img[:,:,1].shape)
img =np.array([np.pad(img[:,:,0], (3,3), 'constant', constant_values=(200,200)),np.pad(img[:,:,1], (3,3), 'constant',constant_values=(0,0)),np.pad(img[:,:,2], (3,3), 'constant',constant_values=(0,0))])

print(img.shape)
plt.imshow(img.transpose(1,2,0))
plt.imshow(Image.open(test_pairs[299][1][0]))
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.imshow(Image.open(test_pairs[5][1][0]))

import math
print(1-(math.factorial(1999)/(math.factorial(1999-20))*math.pow(1/2000.,20)))
print(20/len(test_pairs))

