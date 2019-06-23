# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
import json
import numpy as np
from keras.models import Model
from keras import regularizers
from tool.utils import filter_num
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from data.load_data import out_data, compute
from keras.layers.core import RepeatVector
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Add, LeakyReLU


def conv_block(input_name_list, input_shape_list, filter_factor, ratio, bias=1, mode=None, attention=True):
    output_list = []
    conv_list = []
    dropout_list = []
    max_list = []
    avg_list = []
    dense_list = []
    max_dense_list = []
    avg_dense_list = []
    dense_output_list = []
    max_output_list = []
    avg_output_list = []
    add_list = []
    repeat_list = []
    multiply_list = []
    pool_list = []
    flatten_list = []
    for i in range(len(input_name_list)):
        base = str(filter_factor) + '_' + str(i)
        if mode == 'input':
            input_name_list[i] = Input(shape=input_shape_list[i])
        # add layer name
        conv_list.append('conv_' + base)
        dropout_list.append('dropout_'+base)
        pool_list.append('pool_' + base)
        output_list.append('output_' + base)
        if attention is True:
            max_list.append('max_'+base)
            avg_list.append('avg_'+base)
            dense_list.append('dense_'+base)
            max_dense_list.append('max_dense_'+base)
            avg_dense_list.append('avg_dense_'+base)
            dense_output_list.append('dense_output_'+base)
            max_output_list.append('max_output_'+base)
            avg_output_list.append('avg_output_'+base)
            add_list.append('add_'+base)
            repeat_list.append('repeat_'+base)
            multiply_list.append('multiply_'+base)
        # add layer
        conv_list[i] = Conv1D(nb_filter=filter_num(input_shape_list[i][0], bias)//filter_factor, filter_length=3,
                              strides=2, padding='same', activation='elu', kernel_initializer='glorot_uniform',
                              bias_initializer='glorot_uniform')(input_name_list[i])
        if mode == 'input':
            dropout_list[i] = Dropout(0)(conv_list[i])
        else:
            dropout_list[i] = Dropout(0.2)(conv_list[i])
        if attention is True:
            max_list[i] = GlobalMaxPooling1D()(dropout_list[i])
            avg_list[i] = GlobalAveragePooling1D()(dropout_list[i])

            dense_list[i] = Dense(count_neurals(filter_num(input_shape_list[i][0], bias), ratio), activation='elu',
                                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
            max_dense_list[i] = dense_list[i](max_list[i])
            avg_dense_list[i] = dense_list[i](avg_list[i])
            dense_output_list[i] = Dense(filter_num(input_shape_list[i][0], bias)//filter_factor, activation='elu',
                                         kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
            max_output_list[i] = dense_output_list[i](max_dense_list[i])
            avg_output_list[i] = dense_output_list[i](avg_dense_list[i])
            add_list[i] = Add()([max_output_list[i], avg_output_list[i]])
            repeat_list[i] = RepeatVector(int(conv_list[i].shape[1]))(add_list[i])
            multiply_list[i] = Multiply()([repeat_list[i], dropout_list[i]])
            pool_list[i] = MaxPooling1D(2, padding='same')(multiply_list[i])
        else:
            pool_list[i] = MaxPooling1D(2, padding='same')(dropout_list[i])
        output_list[i] = BatchNormalization()(pool_list[i])
        if mode == 'output':
            flatten_list.append('flatten_' + str(i))
            # if attention is True:
            #     flatten_list[i] = Flatten()(output_list[i])
            # else:
            flatten_list[i] = Flatten()(output_list[i])
    if mode == 'output':
        return flatten_list
    # if attention is True:
    #     return multiply_list
    return output_list


def count_neurals(a, b):
    if a//b < 2:
        print("The neurals is less than 2, and it been set to 2!")
        return 2
    else:
        return a//b


def build_model(data, para_dict):
    """
    模型
    :param data: dict 模型输入数据
    :param para_dict: dict 模型参数
    :return:
    """
    data_key = list(data['x_test'].keys())
    data_key.sort()
    # data_key = ['CD5', 'CD1', 'CA5', 'CD2', 'CD4', 'CD3']
    # input_shape_list = [(23, 1), (360, 1), (23, 1), (180, 1), (45, 1), (90, 1)]
    print('data_key:', data_key)
    input_shape_list = []
    input_length = []
    for key in data_key:
        input_shape_list.append(data['x_test'][key].shape[1:])
        input_length.append(data['x_test'][key].shape[1])
    # input_shape_list = [(23, 1), (360, 1), (23, 1), (180, 1), (45, 1), (90, 1)]
    print('input_shape_list:', input_shape_list)

    # number = para_dict["attention_layer"]
    # if number == 1:
    #     FIRST = True
    #     SECOND = False
    #     THIRD = False
    # elif number == 2:
    #     FIRST = False
    #     SECOND = True
    #     THIRD = False
    # else:
    #     FIRST = False
    #     SECOND = False
    #     THIRD = True

    FIRST = False
    SECOND = False
    THIRD = True


    length = len(input_shape_list)
    input_list = []
    for i in range(length):
        input_list.append('input_' + str(i))
    first_bn_name_list = conv_block(input_name_list=input_list,
                                    input_shape_list=input_shape_list,
                                    filter_factor=1,
                                    ratio=4,
                                    mode='input',
                                    attention=False)
    second_bn_name_list = conv_block(input_name_list=first_bn_name_list,
                                     input_shape_list=input_shape_list,
                                     filter_factor=2,
                                     ratio=4,
                                     attention=False)
    # third_bn_name_list = conv_block(input_name_list=second_bn_name_list,
    #                                 input_shape_list=input_shape_list,
    #                            2-3layers     filter_factor=4,
    #                                 ratio=4,
    #                                 attention=False)
    # forth_bn_name_list = conv_block(input_name_list=third_bn_name_list,
    #                                 input_shape_list=input_shape_list,
    #                                 filter_factor=4,
    #                                 ratio=4)
    # fifth_bn_name_list = conv_block(input_name_list=forth_bn_name_list,
    #                                 input_shape_list=input_shape_list,
    #                                 filter_factor=16,
    #                                 ratio=4)
    sixth_bn_name_list = conv_block(input_name_list=second_bn_name_list,
                                    input_shape_list=input_shape_list,
                                    filter_factor=4,
                                    ratio=4,
                                    attention=False,
                                    mode='output')

    # output_name_list = []
    # for i in range(length):
    #     output_name_list.append('output_'+str(i))
    #     output_name_list[i] = Flatten()(fifth_bn_name_list[i])
    # concatenate all part of conv module)
    all_input = concatenate(sixth_bn_name_list)
    # dense_1 = Dense(1024, activation='leaky relu', kernel_initializer='glorot_uniform',
    #                 bias_initializer='glorot_uniform')(all_input)
    # dropout_1 = Dropout(0.2)(dense_1)
    # bn_1 = BatchNormalization()(dropout_1)
    dense_2 = Dense(512, activation='elu', kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform')(all_input)
    activation_2 = LeakyReLU(alpha=0.05)(dense_2)
    dropout_2 = Dropout(0.2)(activation_2)
    bn_2 = BatchNormalization()(dropout_2)
    dense_3 = Dense(128, activation='elu', kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform')(bn_2)
    dropout_3 = Dropout(0.2)(dense_3)
    bn_3 = BatchNormalization()(dropout_3)
    # dense_4 = Dense(64, activation='relu')(bn_3)
    # dropout_4 = Dropout(0.2)(dense_4)
    # bn_4 = BatchNormalization()(dropout_4)
    output = Dense(33, activation='softmax', kernel_initializer='glorot_uniform',
                    bias_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.002))(bn_3)

    model = Model(input_list, output)
    # plot_model(model, to_file='/home/lab307/chenbin/model.eps')

    return model


def run_model(model, data, i, weight_path, write_path=""):
    """
    模型训练
    :param model: object 模型
    :param data： dict 实验数据
    :return:
    """
    data_key = list(data['x_train'].keys())
    data_key.sort()
    train_x, val_x, test_x = sep_data(data, data_key)
    rmsprop = RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.1, patience=10, verbose=2, mode='auto')
    tensorboad = TensorBoard(log_dir='/home/lab307/chenbin/data/original_data/identification/log/')
    checkpoint = ModelCheckpoint(filepath='/home/lab307/chenbin/data/original_data/identification/log/'+str(i)+'.h5',
                                 monitor='val_acc', mode='auto', save_best_only=True, save_weights_only=True)

    # plot_model(model, to_file='/home/lab307/chenbin/paper/paper_picture/model.eps',
    #            show_layer_names=False)  # ,rankdir='LR')

    # history = model.fit(x=train_x, y=data['y_train'], batch_size=32, epochs=200,
    #                     verbose=2, validation_data=(val_x, data['y_val']),
    #                     shuffle=True, callbacks=[tensorboad, checkpoint])
    model.load_weights(filepath=weight_path)
    predict_classes = model.predict(test_x, batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    # pprint(matrix)
    if write_path == "":
        pass
    else:
        write_file(write_path, matrix, acc)
    return acc


def sep_data(data, data_key):
    train_x = []
    val_x = []
    test_x = []
    for key in data_key:
        train_x.append(data['x_train'][key])
        val_x.append(data['x_val'][key])
        test_x.append(data['x_test'][key])
    return train_x, val_x, test_x


def write_file(write_path, matrix, acc):
    confusion_list = matrix.tolist()
    g = open(write_path, 'a')
    for j in range(len(confusion_list)):
        main_data = str(j) + '_' + str(j) + ':' + str(confusion_list[j][j])
        other_data = ''
        for k in range(len(confusion_list[j])):
            if k != j and confusion_list[j][k] != 0:
                other_data = other_data + ' ' + str(j) + '_' + str(k) + ':'+str(confusion_list[j][k])
            else:
                pass
        string = main_data + other_data
        g.write(string+'\n')
    g.write('acc: ' + str(acc)+'\n')
    g.close()


def test_model_flow(read_path, weight_path, write_path, para_dict):
    acc_list = []
    lead_list = list(np.arange(12))
    data = out_data(read_path=read_path, lead_list=lead_list, level=5)
    write_file = write_path + "33.txt"
    # try:
    for times in range(3):
        weight_file = weight_path + str(times) + '.h5'
        model = build_model(data, para_dict)
        acc = run_model(model, data, times, weight_file, write_file)
        acc_list.append(str(acc))
    para_string = json.dumps(para_dict, ensure_ascii=False)
    f = open(write_file, 'a')
    f.write(para_string+'\n')
    string = 'acc = [' + ','.join(acc_list) + ']' + '\n'
    f.write(string)
    f.close()
    # except ValueError:
    #     pass


if __name__ == '__main__':
    para_dict = {'sample_seconds': 6, 'train_seconds': 40, 'test_seconds': 40, 'train_samples': 100,
                 'test_samples': 100, 'val_samples': 50, 'attention_layer': 3}
    original_path = '/home/lab307/chenbin/data/original_data/identification/mit_original_data/'
    sample_path = '/home/lab307/chenbin/data/original_data/identification/train_model/'
    log_path = '/home/lab307/chenbin/data/original_data/identification/log/'
    write_path = '/home/lab307/chenbin/data/original_data/identification/result/'
    test_model_flow(sample_path, log_path, write_path, para_dict)