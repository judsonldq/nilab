# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model
from keras import regularizers
from keras.utils.conv_utils import conv_output_length
from keras.utils.layer_utils import print_summary
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from data.load_data import out_data, compute
from tool.utils import filter_num
from pprint import pprint
from keras.utils import plot_model


def build_model(data, para_dict={}):
    """
    模型
    :param data: dict 模型输入数据
    :param para_dict: dict 模型参数
    :return:
    """
    data_key = list(data['x_train'].keys())
    input_shape_list = []
    input_length = []
    for key in data_key:
        input_shape_list.append(data['x_train'][key].shape[1:])
        input_length.append(data['x_train'][key].shape[1])

    length = len(input_shape_list)
    # the first conv module
    input_name_list = []
    first_conv_name_list = []
    first_pool_name_list = []
    first_bn_name_list = []
    for i in range(length):
        input_name_list.append('input_'+str(i))
        first_conv_name_list.append('first_conv_'+str(i))
        first_pool_name_list.append('first_pool_'+str(i))
        first_bn_name_list.append('first_bn_'+str(i))
        input_name_list[i] = Input(input_shape_list[i])
        first_conv_name_list[i] = Conv1D(nb_filter=filter_num(input_shape_list[i][0]), filter_length=3, strides=2,
                                         padding='same', activation='relu')(input_name_list[i])
        # config = layer_from_config()
        # first_conv_name_list[i] = layer_from_config(config)
        # print(first_conv_name_list[i].output_shape)
        # print_summary(model)
        first_pool_name_list[i] = MaxPooling1D(2, padding='same')(first_conv_name_list[i])
        first_bn_name_list[i] = BatchNormalization()(first_pool_name_list[i])

    second_bn_name_list = conv_block(first_bn_name_list, input_shape_list, 2)
    third_bn_name_list = conv_block(second_bn_name_list, input_shape_list, 4)
    forth_bn_name_list = conv_block(third_bn_name_list, input_shape_list, 8)
    fifth_bn_name_list = conv_block(forth_bn_name_list, input_shape_list, 16)
    # # the second conv module
    # second_conv_name_list = []
    # second_pool_name_list = []
    # second_bn_name_list = []
    # for i in range(length):
    #     second_conv_name_list.append('second_conv_'+str(i))
    #     second_pool_name_list.append('second_pool_'+str(i))
    #     second_bn_name_list.append('second_bn_'+str(i))
    #     second_conv_name_list[i] = Conv1D(nb_filter=filter_num(input_shape_list[i][0])//2, filter_length=3, strides=2,
    #                                       padding='same', activation='relu')(first_bn_name_list[i])
    #     second_pool_name_list[i] = MaxPooling1D(2)(second_conv_name_list[i])
    #     second_bn_name_list[i] = BatchNormalization()(second_pool_name_list[i])
    # the third conv module

    output_name_list = []
    last_conv_name_list = []
    last_pool_name_list = []
    last_bn_name_list = []
    for i in range(length):
        output_name_list.append('output_'+str(i))
        if input_shape_list[i][0] <= 128:
            last_conv_name_list.append(None)
            last_pool_name_list.append(None)
            last_bn_name_list.append(None)
            output_name_list[i] = Flatten()(fifth_bn_name_list[i])
        else:
            last_conv_name_list.append('third_conv_'+str(i))
            last_pool_name_list.append('third_pool_'+str(i))
            last_bn_name_list.append('third_bn_'+str(i))
            last_conv_name_list[i] = Conv1D(nb_filter=filter_num(input_shape_list[i][0])//16, filter_length=3, strides=2,
                                             padding='same', activation='relu')(fifth_bn_name_list[i])
            last_pool_name_list[i] = MaxPooling1D(2, padding='same')(last_conv_name_list[i])
            last_bn_name_list[i] = BatchNormalization()(last_pool_name_list[i])
            output_name_list[i] = Flatten()(last_bn_name_list[i])
    # concatenate all part of conv module
    all_input = concatenate(output_name_list)
    dense_1 = Dense(512, activation='relu')(all_input)
    dropout_1 = Dropout(0.2)(dense_1)
    bn_1 = BatchNormalization()(dropout_1)
    dense_2 = Dense(256,activation='relu')(bn_1)
    dropout_2 = Dropout(0.2)(dense_2)
    bn_2 = BatchNormalization()(dropout_2)
    dense_3 = Dense(128, activation='relu')(bn_2)
    dropout_3 = Dropout(0.2)(dense_3)
    bn_3 = BatchNormalization()(dropout_3)
    dense_4 = Dense(64, activation='relu')(bn_3)
    dropout_4 = Dropout(0.2)(dense_4)
    bn_4 = BatchNormalization()(dropout_4)
    output = Dense(50, activation='softmax', kernel_regularizer=regularizers.l2(0.00))(bn_4)

    model = Model(input_name_list, output)
    return model


def conv_block(input_name_list, input_shape_list, filter_factor):
    output_name_list = []
    conv_name_list = []
    pool_name_list = []
    for i in range(len(input_name_list)):
        base = str(filter_factor) + '_' + str(i)
        output_name_list.append('output_' + base)
        conv_name_list.append('conv_' + base)
        pool_name_list.append('pool_' + base)
        conv_name_list[i] = Conv1D(nb_filter=filter_num(input_shape_list[i][0])//filter_factor, filter_length=3,
                                   strides=2, padding='same', activation='relu')(input_name_list[i])
        pool_name_list[i] = MaxPooling1D(2, padding='same')(conv_name_list[i])
        output_name_list[i] = BatchNormalization()(pool_name_list[i])
    return output_name_list


def run_model(model, data, write_path=""):
    """
    模型训练
    :param model: object 模型
    :param data： dict 实验数据
    :return:
    """
    data_key = list(data['x_train'].keys())
    train_x, val_x, test_x = sep_data(data, data_key)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0050, patience=5, verbose=2, mode='auto')
    plot_model(model, to_file='/home/lab307/chenbin/paper/paper_picture/model.eps',
               show_layer_names=False)  # ,rankdir='LR')

    history = model.fit(x=train_x, y=data['y_train'], batch_size=32, epochs=200,
                        verbose=2, validation_data=(val_x, data['y_val']),
                        shuffle=True)#, callbacks=[early_stopping])
    predict_classes = model.predict(test_x, batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    matrix = matrix.astype('str')
    # pprint(matrix)
    if write_path == "":
        pass
    else:
        f = open(write_path, 'a')
        f.write('confusion matrix:' + '\n')
        for i in range(len(matrix)):
            f.write('\t'.join(matrix[i]) + '\n')
        f.write('acc:' + str(acc) + '\n')
        f.close()
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


def model_flow(read_path, write_path):
    acc_list = []
    lead_list = list(np.arange(12))
    data = out_data(read_path=read_path, lead_list=lead_list, level=5)
    write_file = write_path + "increase_training_samples.txt"
    for times in range(1):
        model = build_model(data)
        acc = run_model(model, data, write_file)
        acc_list.append(str(acc))
    f = open(write_file, 'a')
    string = 'acc_2-3layers = [' + ','.join(acc_list) + ']' + '\n'
    f.write(string)
    f.close()



if __name__ == '__main__':
    read_path = '/home/lab307/chenbin/data/experiment/identification/'
    write_path = '/home/lab307/chenbin/paper/experiment/identification/'
    model_flow(read_path, write_path)