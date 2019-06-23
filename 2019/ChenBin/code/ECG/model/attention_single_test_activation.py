# -*- encoding:utf-8 -*-

import sys
sys.path.append('./,../')
import json
from tool.helper import filter_num
from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from model.load_data import out_data, compute
from keras.layers.core import RepeatVector, Activation
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization,\
    GlobalAveragePooling1D, GlobalMaxPooling1D, Multiply, Add


def run_model(write_path, data, para_dict):
    single_length = data['x_train']['single_beat'].shape[1:]
    ratio = para_dict['ratio']
    min_delta = para_dict['min_delta']
    bias = para_dict['bias']
    epoch = para_dict['epoch']
    input_beat = Input(shape=single_length)
    input_name_list = [input_beat]
    first_block = conv_block(input_name_list=input_name_list, input_shape=single_length, ratio=ratio,
                             name='beat', filter_factor=1, bias=bias, attention=False, mode='input')
    # second_block = conv_block(input_name_list=first_block, input_shape=single_length, ratio=ratio,
    #                           name='beat', filter_factor=2, bias=bias, attention=False)
    third_block = conv_block(input_name_list=first_block, input_shape=single_length, ratio=ratio,
                             name='beat', filter_factor=4, bias=bias, mode='output')
    # 全连接层
    dense_1 = Dense(64, activation='relu')(third_block[0])
    dropout_1 = Dropout(0.2)(dense_1)
    bn_1 = BatchNormalization()(dropout_1)
    dense_2 = Dense(16, activation='relu')(bn_1)
    dropout_2 = Dropout(0.2)(dense_2)
    bn_2 = BatchNormalization()(dropout_2)
    output = Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(bn_2)

    model = Model(inputs=input_beat, outputs=output)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=4, verbose=2, mode='auto')

    history = model.fit(x=data['x_train']['single_beat'], y=data['y_train'], batch_size=32, epochs=epoch, verbose=2,
                        validation_data=(data['x_val']['single_beat'], data['y_val']),
                        shuffle=True, callbacks=[early_stopping])
    predict_classes = model.predict(data['x_test']['single_beat'], batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    matrix = matrix.astype('str')
    f = open(write_path, 'a')
    f.write('confusion matrix:' + '\n')
    for i in range(len(matrix)):
        f.write('\t'.join(matrix[i]) + '\n')
    f.write('acc:' + str(acc) + '\n')
    f.close()
    return acc, model


def conv_block(input_name_list, input_shape, name, filter_factor, ratio, bias=0, mode=None, attention=True):
    output_list = []
    conv_list = []
    dropout_list = []
    pool_list = []
    flatten_list = []
    dropout_para = 0.2
    base = name + str(filter_factor)
    if mode == 'input':
        dropout_para = 0
    # add layer name
    conv_list.append('conv_' + base)
    dropout_list.append('pool_' + base)
    pool_list.append('pool_' + base)
    output_list.append('output_' + base)
    # conv layer
    conv_list[0] = Conv1D(nb_filter=filter_num(input_shape[0], bias) // filter_factor, filter_length=3,
                          strides=2, padding='same', activation='relu')(input_name_list[0])
    dropout_list[0] = Dropout(dropout_para)(conv_list[0])
    if attention is True:
        attention_output = attention_module(dropout_list[0], ratio, base)
        pool_list[0] = MaxPooling1D(2, padding='same')(attention_output)
    else:
        pool_list[0] = MaxPooling1D(2, padding='same')(dropout_list[0])
    output_list[0] = BatchNormalization()(pool_list[0])
    if mode == 'output':
        flatten_list.append('flatten' + base)
        flatten_list[0] = Flatten()(output_list[0])
    if mode == 'output':
        return flatten_list
    return output_list


def attention_module(input_name, ratio, base):
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
    activation_list = []
    multiply_list = []
    max_list.append('max_'+base)
    avg_list.append('avg_'+base)
    dense_list.append('dense_'+base)
    max_dense_list.append('max_dense_'+base)
    avg_dense_list.append('avg_dense_'+base)
    dense_output_list.append('dense_output_'+base)
    max_output_list.append('max_output_'+base)
    avg_output_list.append('avg_output_'+base)
    add_list.append('add_'+base)
    activation_list.append('act_'+base)
    repeat_list.append('repeat_'+base)
    multiply_list.append('multiply_'+base)
    # add layer
    # print(input_name.shape)
    max_list[0] = GlobalMaxPooling1D()(input_name)
    avg_list[0] = GlobalAveragePooling1D()(input_name)
    filter_num = int(input_name.shape[-1]) //ratio
    if filter_num < 2:
        print('THE ENCODE INPUT LAYER FILTER NUMBER IS SMALL THAN 2!!!! HAVE RESET IT TO 2!!!!')
        filter_num = 2
    dense_list[0] = Dense(filter_num, activation='relu')
    max_dense_list[0] = dense_list[0](max_list[0])
    avg_dense_list[0] = dense_list[0](avg_list[0])
    dense_output_list[0] = Dense(int(input_name.shape[-1]), activation='relu')
    max_output_list[0] = dense_output_list[0](max_dense_list[0])
    avg_output_list[0] = dense_output_list[0](avg_dense_list[0])
    add_list[0] = Add()([max_output_list[0], avg_output_list[0]])
    activation_list[0] = Activation(activation='relu')(add_list[0])
    repeat_list[0] = RepeatVector(int(input_name.shape[1]))(activation_list[0])
    multiply_list[0] = Multiply()([repeat_list[0], input_name])
    return multiply_list[0]


def attention_experiment(para_dict):
    read_path = '../beat/'
    acc_file_path = './single_activation.txt'
    acc_list = []
    data = out_data(read_path=read_path)
    for times in range(para_dict['times']):
        acc, model = run_model(acc_file_path, data, para_dict)
        acc_list.append(str(acc))
        # model.save_weights('../weight/'
        #                    + str(times) + '.h5')
    f = open(acc_file_path, 'a')
    para_string = json.dumps(para_dict)
    f.write(para_string+'\n')
    string = 'acc = [' + ','.join(acc_list) + ']' + '\n'
    f.write(string)
    f.close()


if __name__ == "__main__":
# def single():
    para_dict = {'min_delta': 0.002, 'ratio': 4, 'bias': 0, 'epoch': 50, 'times': 10, 'patience': 4}
    min_delta_list = [0.002, 0.005, 0.02]
    for i in range(len(min_delta_list)):
        para_dict['min_delta'] = min_delta_list[i]
        attention_experiment(para_dict)



