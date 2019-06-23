# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Time:     2018/03/09 15:00
@Author:   ChenBin
@Function: build cnn model by using a function
"""
import numpy as np
import os
import h5py
from random import shuffle
import pandas as pd
from keras.layers import Input,Conv1D,MaxPooling1D,Flatten,Dense,Dropout,BatchNormalization
from keras.layers import concatenate
from keras.models import Model
from keras.utils import np_utils,plot_model
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt


def get_data(path,files):
    x = []
    y = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path,files),'r')
    for lines in f.readlines():
        temp = lines.strip().split(',')
        floatData = []
        for i in range(len(temp)):
            floatData.append(float(temp[i]))
        x.append(floatData)
    return np.array(x),np.array(y)
def all_data(path):
    global nb_classes
    X = []
    Y = []
    fileList = os.listdir(path)
    for files in fileList:
        x,y = get_data(path,files)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype('float32')
    #X = np.expand_dims(X, axis=2)#binary need axis = 2
    Y = Y.astype('float32')
    Y = np_utils.to_categorical(Y, nb_classes)
    #print ('run to here!')
    return X,Y
def compute_acc(predict_classes,y_test):
    y_predict = []
    y_true = []
    for i in range(len(y_test)):
        y_true.append(list(y_test[i]).index(max(y_test[i])))
    for i in range(len(predict_classes)):
        y_predict.append(list(predict_classes[i]).index(max(predict_classes[i])))
    matrix = confusion_matrix(y_true, y_predict)
    matrix = pd.DataFrame(matrix)
    acc = accuracy_score(y_true, y_predict)
    return matrix, acc

train_path = '/home/chenbin/data_process/experiment/5/train/'
val_path = '/home/chenbin/data_process/experiment/5/val/'
test_path = '/home/chenbin/data_process/experiment/5/test/'
train_fileList = os.listdir(train_path)
train_dic = {}
train_Label = []
for every in train_fileList:
    getLabel = int((every.split('_')[-1]).split('.')[0])
    if getLabel not in train_Label:
        train_Label.append(getLabel)
train_Label = sorted(train_Label)
for i in range(len(train_Label)):
    train_dic[train_Label[i]] = i

input_shape = (256, 1)
kernel_size = 3
maxPooling_size = 2
nb_classes = len(train_dic)
nb_epoch = 50
batch_size = 64
generator_size = 20
max_q_size = 200
train_samples_per_epoch = len(train_fileList)

#combine different scare in the second CNN layers
input_ecg = Input(shape=input_shape)

encode_0 = BatchNormalization()(input_ecg)
encode_1 = Conv1D(nb_filter=32,filter_length=kernel_size,strides=2,padding='same',activation='relu')(encode_0)
dropout_0 = Dropout(0.1)(encode_1)
encode_2 = MaxPooling1D(maxPooling_size)(dropout_0)
encode_3 = BatchNormalization()(encode_2)
encode_4 = Conv1D(nb_filter=64,filter_length=kernel_size,strides=2,padding='same',activation='relu')(encode_3)
dropout_1 = Dropout(0.2)(encode_4)
encode_5 = MaxPooling1D(maxPooling_size)(dropout_1)
encode_6 = BatchNormalization()(encode_5)
encode_7 = Conv1D(nb_filter=64,filter_length=kernel_size,strides=2,padding='same',activation='relu')(encode_6)
dropout_2 = Dropout(0.2)(encode_7)
encode_8 = MaxPooling1D(maxPooling_size)(dropout_2)
encode_9 = BatchNormalization()(encode_8)
encode_10 = Flatten()(encode_9)

dense = Dense(64,activation='relu')(encode_10)
dropout = Dropout(0.4)(dense)
output = Dense(nb_classes,activation='softmax')(dropout)

model = Model(input_ecg,output)

rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-06)
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
EarlyStopping(monitor='val_loss', min_delta=0.0050, patience=4, verbose=1, mode='auto')

x_train,y_train = all_data(train_path)
x_val,y_val = all_data(val_path)
x_test,y_test = all_data(test_path)


accFlag = 0.0
for i in range(10):
    history = model.fit(x = x_train,y = y_train,batch_size=batch_size,epochs=5,
                        verbose=1,validation_data = (x_val,y_val),shuffle=True)
    print ('predict process:')
    predict_classes = model.predict(x_test,batch_size = batch_size, verbose = 1)
    print ('\n'+'confusion matrix:')
    matrix,acc = compute_acc(predict_classes,y_test)
    if acc > accFlag:
        accFlag = acc
        #model.save_weights('/media/lab307/aa9d6aa6-56d4-49e0-80cd-908035f93743/chen/model/try/try.h5')
    print (matrix)
    print (acc)