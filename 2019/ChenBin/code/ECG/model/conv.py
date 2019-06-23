# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Time:     2018/03/06 20:40
@Author:   ChenBin
@Function: build a 1D-cnn _model with three conv layers and two dense layers
"""

import os
import re
import json
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.models import Model, model_from_json
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization


class CNN(object):

    def __init__(self, nb_classes=5, input_size=(256, 1), nb_epoch=50, batch_size=8, verbose=1,
                 kernel_size={'C1': 3, 'C2': 3, 'C3': 3}, nb_num={'C1': 16, 'C2': 32, 'C3': 64, 'D1': 15},
                 strides={'C1': 1, 'C2': 1, 'C3': 1}, padding={'C1': 'valid', 'C2': 'valid', 'C3': 'valid'},
                 pool_size={'C1': 2, 'C2': 2, 'C3': 2}, dropout={'C1': 0.0, 'C2': 0.0, 'C3': 0.2, 'D1': 0.2},
                 optimizer={'lr': 0.1, 'rho': 0.9, 'epsilon': 1e-06},
                 early_stop={'monitor': 'val_loss', 'min_delta': 0.0050, 'patience': 4, 'verbose': 1, 'mode': 'auto'}):
        """

        :param nb_classes: int 数据源的类别总数
        :param input_size: tuple 输入数据的格式
        :param nb_epoch: int 模型训练次数
        :param batch_size: int 每个batch输入样本个数
        :param verbose: int 训练进度显示参数
        :param kernel_size: dict 卷积核的大小
        :param nb_num: dict 每层神经元的数量，包括卷积层和全连接层
        :param strides: dict 每层步长
        :param padding: dict 每层边缘填充 可选'vaild' 'same' 'causal'
        :param pool_size: dict 各池化层池化尺寸
        :param dropout: dict 各dropout层参数
        :param optimizer: dict 优化器参数
        :param early_stop: dict earlyStopping层参数
        """
        self.model_name = '1D-CNN'
        self._param = {'nb_classes': nb_classes, 'input_size': input_size, 'nb_epoch': nb_epoch, 'verbose': verbose,
                       'batch_size': batch_size, 'kernel_size': kernel_size, 'nb_num': nb_num,
                       'strides': strides, 'padding': padding, 'pool_size': pool_size, 'dropout': dropout,
                       'optimizer': optimizer, 'early_stop': early_stop}

    def get_params(self):
        return self._param

    def set_params(self, params):
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
            self._param[key] = value
        return self

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        模型训练函数
        :param x_train: np.array 训练集
        :param y_train: np.array 训练集标签
        :param x_val: np.array 验证集
        :param y_val: np.array 验证集标签
        :return: history: object 训练过程对象
                 model: object 模型对象
        """
        model = self.model_building()
        rmsprop = RMSprop(lr=self._param['optimizer']['lr'], rho=self._param['optimizer']['rho'],
                          epsilon=self._param['optimizer']['epsilon'])
        model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
        EarlyStopping(monitor=self._param['early_stop']['monitor'], min_delta=self._param['early_stop']['min_delta'],
                      patience=self._param['early_stop']['patience'], verbose=self._param['early_stop']['verbose'],
                      mode=self._param['early_stop']['mode'])
        start = time.time()
        history = model.fit(x=x_train, y=y_train, batch_size=self._param['batch_size'], epochs=self._param['nb_epoch'],
                            verbose=self._param['verbose'], validation_data=(x_val, y_val), shuffle=True)
        end = time.time()
        print('模型训练时间为：', str(end - start))
        return history, model

    # def model_building(self):
    #     input_ecg = Input(shape=(128, 1))
    #
    #     encode_0 = BatchNormalization()(input_ecg)
    #     encode_1 = Conv1D(nb_filter=32, filter_length=3, strides=2, padding='same', activation='relu')(
    #         encode_0)
    #     dropout_0 = Dropout(0.1)(encode_1)
    #     encode_2 = MaxPooling1D(2)(dropout_0)
    #     encode_3 = BatchNormalization()(encode_2)
    #     encode_4 = Conv1D(nb_filter=64, filter_length=3, strides=2, padding='same', activation='relu')(
    #         encode_3)
    #     dropout_1 = Dropout(0.2)(encode_4)
    #     encode_5 = MaxPooling1D(2)(dropout_1)
    #     encode_6 = BatchNormalization()(encode_5)
    #     encode_7 = Conv1D(nb_filter=64, filter_length=3, strides=2, padding='same', activation='relu')(
    #         encode_6)
    #     dropout_2 = Dropout(0.2)(encode_7)
    #     encode_8 = MaxPooling1D(2)(dropout_2)
    #     encode_9 = BatchNormalization()(encode_8)
    #     encode_10 = Flatten()(encode_9)
    #
    #     dense = Dense(64, activation='relu')(encode_10)
    #     dropout = Dropout(0.4)(dense)
    #     output = Dense(5, activation='softmax')(dropout)
    #
    #     model = Model(input_ecg, output)
    #     return model
    def model_building(self):
        """
        搭建模型
        :return: cnn_model: object 返回搭建的模型对象
        """
        input_ecg = Input(shape=self._param['input_size'])
        encode_0 = BatchNormalization()(input_ecg)
        encode_1 = Conv1D(nb_filter=self._param['nb_num']['C1'], filter_length=self._param['kernel_size']['C1'],
                          strides=self._param['strides']['C1'], padding=self._param['padding']['C1'],
                          activation='relu')(encode_0)
        dropout_0 = Dropout(rate=self._param['dropout']['C1'])(encode_1)
        encode_2 = MaxPooling1D(self._param['pool_size']['C1'])(dropout_0)
        encode_3 = BatchNormalization()(encode_2)
        encode_4 = Conv1D(nb_filter=self._param['nb_num']['C2'], filter_length=self._param['kernel_size']['C2'],
                          strides=self._param['strides']['C2'], padding=self._param['padding']['C2'],
                          activation='relu')(encode_3)
        dropout_1 = Dropout(rate=self._param['dropout']['C2'])(encode_4)
        encode_5 = MaxPooling1D(self._param['pool_size']['C2'])(dropout_1)
        encode_6 = BatchNormalization()(encode_5)
        encode_7 = Conv1D(nb_filter=self._param['nb_num']['C3'], filter_length=self._param['kernel_size']['C3'],
                          strides=self._param['strides']['C3'], padding=self._param['padding']['C3'],
                          activation='relu')(encode_6)
        dropout_2 = Dropout(rate=self._param['dropout']['C3'])(encode_7)
        encode_8 = MaxPooling1D(self._param['pool_size']['C3'])(dropout_2)
        encode_9 = BatchNormalization()(encode_8)
        encode_10 = Flatten()(encode_9)
        dense = Dense(units=self._param['nb_num']['D1'], activation='relu')(encode_10)
        dropout = Dropout(rate=self._param['dropout']['D1'])(dense)
        output = Dense(self._param['nb_classes'], activation='softmax')(dropout)
        cnn_model = Model(input_ecg, output)
        return cnn_model

    def predict(self, x_test, model):
        """
        模型预测函数
        :param x_test: np.array 测试集
        :param model: object 模型对象
        :return: y_predict: list 预测结果
        """
        y_predict = model.predict(x_test, batch_size=32, verbose=1)
        y = []
        for classes in y_predict:
            classes = list(classes)
            y.append(classes.index(max(classes)))
        return y

    def save_model(self, model, save_path):
        """
        存储模型
        :param model: object 训练好的模型
        :param save_path: string 模型存储路径
        :return: None
        """
        # 判断当前存储的model和weight文件的序号
        file_list = os.listdir(save_path)
        version = 0
        if len(file_list) > 0:
            for files in file_list:
                if re.search(r'model', files) is not None:
                    temp = int(files.split('.')[0].split('_')[-1])
                    if temp > version:
                        version = temp + 1
        # 路径补全
        weight_path = save_path + 'weight_' + str(version) + '.h5'
        model_path = save_path + 'model_' + str(version) + '.json'
        note_path = save_path + 'note.txt'
        print('模型version：', str(version))
        model_json = model.to_json()
        open(model_path, 'w').write(model_json)
        print('模型结构存储完毕，存储位置：', model_path)
        model.save_weights(weight_path)
        print('网络权重存储完毕，存储位置：', weight_path)
        note_dict = {'model_name': 'model_' + str(version), 'model_feature': self._param}
        json.dump(note_dict, open(note_path, 'a'))
        print('模型注释存储完毕，存储位置：', note_path)

    def load_model(self, load_path, version):
        """
        载入模型
        :param load_path: string 载入模型路径
        :param version: int 载入模型版本
        :return: model: object 模型对象
        """
        model_path = load_path + 'model_' + str(version) + '.json'
        weight_path = load_path + 'weight_' + str(version) + '.h5'
        note_path = load_path + 'note.txt'
        model = model_from_json(open(model_path, 'r').read())
        model.load_weights(weight_path)
        print('模型载入完成！')
        lines = json.loads(open(note_path, 'r').readlines()[-1])
        print('模型注释字典：')
        print(lines)
        return model

    def draw_history(self, history):
        """
        绘制训练过程acc与loss
        :param history: object 训练过程对象
        :return: None
        """

        train_acc = history.history['acc']
        train_loss = history.history['loss']
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']
        length = len(train_acc)
        x = list(range(1, length + 1))
        fig = plt.figure()
        # 绘制loss曲线
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(x, train_loss, 'blue')
        ax1.plot(x, val_loss, 'blue')
        ax1.set_ylabel('loss')
        ax1.set_title('The process of training.')
        # 绘制双y轴曲线图
        ax2 = ax1.twinx()
        # 绘制acc曲线
        ax2.plot(x, train_acc, 'red')
        ax2.plot(x, val_acc, 'red')
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('training times')
        plt.show()

    def confusion_matrix(self, y_true, y_predict):
        """
        计算混淆矩阵
        :param y_true: list 真实标签
        :param y_predict: list 预测标签
        :return: conf_mat: list of pairs 混淆矩阵
        """
        # x轴预测类别 y轴真实类别
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_predict)
        return conf_mat

    def classification_report(self, y_true, y_predict):
        """
        计算每个类别准确率，召回率，F值和宏平均值
        :param y_true: list 真实标签
        :param y_predict: list 预测标签
        :return: clas_rep: 分类报告
        """
        clas_rep = classification_report(y_true=y_true, y_pred=y_predict)
        return clas_rep

    def compute_acc(self, y_true, y_predict):
        """
        计算准确率
        :param y_true: list 真实标签
        :param y_predict: list 预测标签
        :return: acc: float 准确率
        """
        num = 0
        for i in len(y_predict):
            if y_predict[i] == y_true[i]:
                num += 1
        acc = round((num/(len(y_predict)*1.0)), 4)
        return acc

    def draw_heat_map(self, array):
        """
        画热度图
        :param array: list of pairs 混淆矩阵
        :return: None
        """
        cmap = cm.Blues
        figure = plt.figure(facecolor='w')
        ax = figure.add_subplot(1, 1, 1)
        ax.set_yticks(range(len(array)))
        ax.set_xticks(range(len(array)))
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predict Label')
        ax.set_yticklabels(range(len(array)))
        ax.set_xticklabels(range(len(array)))
        max_num = array[0][0]
        min_num = array[0][0]
        for i in array:
            for j in i:
                if j > max_num:
                    max_num = j
                if j < min_num:
                    min_num = j
        heat_map = ax.imshow(array, interpolation='nearest', cmap=cmap, aspect='auto', vmin=min_num, vmax=max_num)
        cb = plt.colorbar(mappable=heat_map, cax=None, ax=None, shrink=0.5)
        plt.show()
