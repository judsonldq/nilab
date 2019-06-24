# -*- coding: utf-8 -*-
import pickle

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

import pandas as pd
import numpy as np
import math
import time

from keras import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers
from keras.models import load_model
from keras import backend as K
from sklearn import metrics
from sklearn.metrics import mean_squared_error
#带lstm的attention模型
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
# from attention_utils import get_activations, get_data_recurrent
from keras.layers import Multiply
"""
输入广告／非广告特征向量的pickle文件，进行特征融合
将融合的特征使用SVM进行训练并保存
"""


f = open("features-audio-not.pkl", "rb")# 读取特征向量的pickle文件
audio = pickle.load(f)
f.close()
f = open("features-audio-ad.pkl", "rb")
audio_ad = pickle.load(f)
f.close()
f = open("features-image-not.pkl", "rb")
image = pickle.load(f)
f.close()
f = open("features-image-ad.pkl", "rb")
image_ad = pickle.load(f)
f.close()
image.extend(image_ad)
audio.extend(audio_ad)

for i in range(len(audio)):
    image[i].extend(audio[i])

labels = [0.0]*(len(image)-len(image_ad))# 根据广告非广告的数据长度分好标签
labels.extend([1.0]*len(image_ad))


# labels = [0.0]*(len(image)-len(image_ad))# 根据广告非广告的数据长度分好标签
# labels.extend([1.0]*len(image_ad))
#
# image_mat = []
# audio_mat = []
# for item in image:
#     if len(item) != 4:
#         print("Error")
#     image_mat.extend(item)
# for item in audio:
#     audio_mat.extend(item)
# pca = PCA(n_components=84)
# image_84 = pca.fit_transform(image_mat)# 进行特征降维度2048降到84维

#
# def mat_to_vec(features):# 将特征矩阵合并成一个向量，取2048X4维向量的均值变成2048X1维
#     features_vec = []
#     for i in range(len(labels)):
#         # concatenate 4 feature vectors as one
#         # features_vec.append(np.concatenate(features_300[4*i:4*i+4]))
#
#         # mean 4 feature vectors as one
#         features_vec.append(np.mean(features[4 * i:4 * i + 4], axis=0))
#
#         # sum 4 feature vectors as one
#         # features_vec.append(np.sum(features_300[4*i:4*i+4],axis=0))
#     return features_vec


# image_vec = mat_to_vec(image_mat)
# image_vec = np.array(image_vec)
# # audio_84 = pca.fit_transform(audio_mat)
# audio_vec = np.array(audio_mat)
#
# feature = np.hstack([image_vec, audio_vec]) # 直接拼接图片和音频的特征向量
# feature=feature.tolist()
# norm = preprocessing.normalize(feature)# 进行归一化

train_data=[]
train_y=[]
train_data = image[:2200]
train_y=labels[:2200]
train_data.extend(image[2669:5100])
train_y.extend(labels[2669:5100])
train_data=np.array(train_data)
train_data= train_data.reshape(train_data.shape[0],5,2048)
print(train_data.shape)

test_data=[]
test_y=[]
test_data = image[2200:2669]
test_y=labels[2200:2669]
test_data.extend(image[5100:5549])
test_y.extend(labels[5100:5549])
test_data=np.array(test_data)
test_data = test_data.reshape(test_data.shape[0],5,2048)
print(test_data.shape)
train_label=train_y
test_label=test_y
#
def buildlstm():

    import numpy as np

    model = Sequential()
    model.add(LSTM(z, input_shape = (5,2048), return_sequences=True))
    model.add(LSTM(z1, input_shape = (5,2048), return_sequences=True))
    model.add(LSTM(zz, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return  model

if __name__ == '__main__':
    z = 32
    z1 = 64
    zz = 16
    model = buildlstm()
    # train_data = np.array(train_data).reshape(-1,13,1)
    # score=runTrain(model, np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label) )
    nbatch_size = 8
    model.fit(np.array(train_data), np.array(train_label),  batch_size= nbatch_size, epochs= 50)
    score = model.evaluate(np.array(test_data), np.array(test_label), batch_size=nbatch_size)
    print(score)
