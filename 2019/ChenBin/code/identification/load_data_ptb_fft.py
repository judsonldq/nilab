# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


def out_data(read_path, lead_list):
    """
    获取数据
    :param read_path: string 数据读取路径
    :param lead_list: list 支路号，用数值代替 0-11
    :param level: int 小波变换阶数
    :return:
    """
#     classes = 251
    train_path = read_path + 'train/'
    test_path = read_path + 'test/'
    val_path = read_path + 'val/'
    x_train, y_train = read_data(train_path, lead_list)
    x_val, y_val = read_data(val_path, lead_list)
    x_test, y_test = read_data(test_path, lead_list, 'test')
    return {'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

def require_train_data(read_path,lead_list):
    train_path = read_path + 'train/'
    x_train, y_train = read_data(train_path, lead_list)
    yield (x_train,y_train)

def require_test_data(read_path,lead_list):
    test_path = read_path + 'test/'
    x_test, y_test = read_data(test_path, lead_list, 'test')
    yield (x_test,y_test)


def read_data(path, lead_list, mode='train'):
    """
    载入数据
    :param path: string 数据集路径
    :param lead_list: list 需要提取的支路号
    :param level: int 小波变换阶数
    :param mode: string 生成训练数据集还是测试数据集
    :return: data_process: np.array 数据集
             label: np.array 数据集对应的标签
    """
    label = []
    data = {}
    file_list = os.listdir(path)
    data['fft'] = []
    data['sta'] = []
    # print(data.keys())
    for files in file_list:
        y = int((files.split('.')[0]).split('_')[-1])
        label.append(y)
        sample_dict = json.load(open(os.path.join(path, files), 'r'))
        sample_key = sample_dict.keys()
        # 取出所有的key，并按照'CAn','CDn','CDn-1'的顺序排序
#         sample_key = list(sample_dict.keys())
#         sample_key.sort(reverse=True)
#         print('sample key :',sample_key)
        # 把'CAn'放到最前面
#         sample_key.insert(0, sample_key.pop())
        for key in sample_key:
            temp = []
            for index in lead_list:
                temp.append(sample_dict[key][index])
            data[key].append(np.array(temp))
    label = np.array(label)
    label = label.astype('float32')
    for key, value in data.items():
        data[key] = np.array(value).astype('float32')
    if mode == 'train':
        label = np.array(np_utils.to_categorical(label))
    return data, label


def get_data(path, files):
    """
    载入单个样本
    :param path: string 样本文件夹路径
    :param files: string 样本文件名
    :return:
    """
    label = int((files.split('.')[0]).split('_')[-1])
    f = open(os.path.join(path, files), 'r')
    data_dict = json.load(f)
    # if len(list(data_dict.keys())) != 6:
    #     print(files)
    # print(data_dict.keys())
    return data_dict, label

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


# if __name__ == "__main__":
#     from pprint import pprint
#     read_path = '/home/lab307/sunhuan/ECG/identification/data/PTB/ptb_ID_continue1/'
#     lead_list = list(np.arange(12))
#     level = 5
#     samples = out_data(read_path, lead_list, level)
#     pprint(samples)