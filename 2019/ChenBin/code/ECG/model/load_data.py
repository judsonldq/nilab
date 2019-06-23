# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from conf.config import KEY_LIST
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


def out_data(read_path, switch=True):
    """
    获取数据
    :param read_path: string 数据读取路径
    :return:
    """
    train_path = read_path + 'train/'
    test_path = read_path + 'test/'
    val_path = read_path + 'val/'
    x_train, y_train = read_data(train_path, 5, switch=switch)
    x_val, y_val = read_data(val_path, 5, switch=switch)
    x_test, y_test = read_data(test_path, 5, switch=switch, mode='test')
    return {'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}


def read_data(path, nb_classes, switch, mode='train', key_list=KEY_LIST):
    """
    载入数据
    :param path: string 数据集路径
    :param nb_classes: int 数据集分类个数
    :param mode: string 生成训练数据集还是测试数据集
    :param key_list: list 文件各行名称
    :return: data_process: np.array 数据集
             label: np.array 数据集对应的标签
    """
    data = {}
    for key in key_list:
        data[key] = []
    label = []
    file_list = os.listdir(path)
    for files in file_list:
        x, y = get_data(path, files)
        for key, value in x.items():
            data[key].append(value)
        label.append(y)
    label = np.array(label)
    label = np.array(label.astype('float32'))
    if mode == 'train':
        label = np.array(np_utils.to_categorical(label, nb_classes))
    samples = {}
    for key, value in data.items():
        value = np.array(value)
        value = value.astype('float32')
        # binary need axis = 2
        if key == 'hand_feature' and switch is False:
            pass
        else:
            value = np.expand_dims(value, axis=2)
        samples[key] = value
    return samples, label


def get_data(path, files):
    """
    载入单个样本
    :param path: string 样本文件夹路径
    :param files: string 样本文件名
    :return:
    """
    label = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path, files), 'r')
    sample = {}
    for lines in f.readlines():
        key, data = lines.strip().split(':')
        data = [float(element) for element in data.split(',')]
        sample[key] = np.array(data)
    return sample, np.array(label)


def compute(y_true, y_predict):
    y = []
    for classes in y_predict:
        classes = list(classes)
        y.append(classes.index(max(classes)))
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y)
    num = 0
    for i in range(len(y)):
        if y[i] == y_true[i]:
            num += 1
    acc = round((num / (len(y) * 1.0)), 4)
    return conf_mat, acc


def trans(multi):
    y = []
    for classes in multi:
        classes = list(classes)
        y.append(classes.index(max(classes)))
    return y
