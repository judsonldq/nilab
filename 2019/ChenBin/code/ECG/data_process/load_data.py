# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time:     2018/03/08 16:10
@Author:   ChenBin
@Function: Processing data_process into model input form
"""

import os
import numpy as np
from keras.utils import np_utils


def load_data(path, nb_classes, mode='train'):
    """
    载入数据
    :param path: string 数据集路径
    :param nb_classes: int 数据集分类个数
    :param mode: string 生成训练数据集还是测试数据集
    :return: data_process: np.array 数据集
             label: np.array 数据集对应的标签
    """
    data = []
    label = []
    file_list = os.listdir(path)
    for files in file_list:
        x, y = get_data(path, files)
        data.append(x)
        label.append(y)
    data = np.array(data)
    label = np.array(label)
    # print(type(data_process), data_process.shape)
    # print(type(data_process[0]), data_process[0])
    data = data.astype('float32')
    # binary need axis = 2
    data = np.expand_dims(data, axis=2)
    label = np.array(label.astype('float32'))
    if mode == 'train':
        label = np.array(np_utils.to_categorical(label, nb_classes))
    return data, label


def get_data(path, files):
    """
    载入单个样本
    :param path: string 样本文件夹路径
    :param files: string 样本文件名
    :return:
    """
    sample = []
    label = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path, files), 'r')
    for lines in f.readlines():
        temp = float(lines.strip())
        sample.append(temp)
    return np.array(sample), np.array(label)


"""
def get_data(path,files):
    x = []
    #print (file)
    y = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path,files),'r')
    for lines in f.readlines():
        temp = lines.strip()
        x.append(float(temp))
    return np.array(x),np.array(y)
def generator_data(fileList,batch_size,path):
    #global sumSample
    global nb_classes
    while 1:
        count = 0
        X = []
        Y = []
        #print ('run to here!')
        for files in fileList:
            x,y = get_data(path,files)
            count += 1
            #if len(x) == 300:
            X.append(x)
            Y.append(y)
            if count == batch_size:
                count = 0
                X = np.array(X)
                X = X.astype('float32')
                X = np.expand_dims(X, axis=2)#binary need axis = 2
                Y = np.array(Y)
                Y = Y.astype('float32')
                #print ('before:',Y)
                #Y = np.expand_dims(Y, -1)
                Y = np_utils.to_categorical(Y, nb_classes)
                #print ('After:',Y)
                yield (X,Y)
                X = []
                Y = []
"""
"""
def load_data(experiment_path,):
    train_path = experiment_path + 'train/'
    test_path = experiment_path + 'test/'
    val_path = experiment_path + 'val/'
    x_train, y_train = load_data(train_path)
    x_val, y_val = load_data(val_path)
    return x_train, x_val, y_train, y_val


def get_data(path,files):
    x = []
    y = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path,files),'r')
    for lines in f.readlines():
        temp = lines.strip()
        x.append(float(temp))
    return np.array(x),np.array(y)


def load_data(path):
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
    X = np.expand_dims(X, axis=2)#binary need axis = 2
    Y = Y.astype('float32')
    Y = np_utils.to_categorical(Y, nb_classes)
    #print ('run to here!')
    return X,Y
"""
