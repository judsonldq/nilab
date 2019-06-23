# -*- encoding:utf-8 -*-

import sys
sys.path.append('./')
from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, \
Dense, Dropout, BatchNormalization,concatenate
from model.load_data import out_data, compute
from keras import backend as K
K.clear_session()


def run_model(write_path, data, min_delta):
    single_length = data['x_train']['single_beat'].shape[1:]
    p_length = data['x_train']['p_wave'].shape[1:]
    qrs_length = data['x_train']['qrs_wave'].shape[1:]
    t_length = data['x_train']['t_wave'].shape[1:]

    # P波部分的三层卷积
    P_part_input = Input(shape=p_length)
    P_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same',
                          activation='relu')(P_part_input)
    P_part_maxpooling_1 = MaxPooling1D(2)(P_part_cnn_1)
    P_part_bn_1 = BatchNormalization()(P_part_maxpooling_1)
    P_part_cnn_2 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same',
                          activation='relu')(P_part_bn_1)
    P_part_dropout_2 = Dropout(0.1)(P_part_cnn_2)
    P_part_maxpooling_2 = MaxPooling1D(2)(P_part_dropout_2)
    P_part_bn_2 = BatchNormalization()(P_part_maxpooling_2)
    P_part_output = Flatten()(P_part_bn_2)

    # QRS波部分的三层卷积
    QRS_part_input = Input(shape=qrs_length)
    QRS_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same',
                            activation='relu')(QRS_part_input)
    QRS_part_maxpooling_1 = MaxPooling1D(2)(QRS_part_cnn_1)
    QRS_part_bn_1 = BatchNormalization()(QRS_part_maxpooling_1)
    QRS_part_cnn_2 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same',
                            activation='relu')(QRS_part_bn_1)
    QRS_part_dropout_2 = Dropout(0.1)(QRS_part_cnn_2)
    QRS_part_maxpooling_2 = MaxPooling1D(2)(QRS_part_dropout_2)
    QRS_part_bn_2 = BatchNormalization()(QRS_part_maxpooling_2)
    QRS_part_output = Flatten()(QRS_part_bn_2)

    # T波部分的三层卷积
    T_part_input = Input(shape=t_length)
    T_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same',
                          activation='relu')(T_part_input)
    T_part_maxpooling_1 = MaxPooling1D(2)(T_part_cnn_1)
    T_part_bn_1 = BatchNormalization()(T_part_maxpooling_1)
    T_part_cnn_2 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same',
                          activation='relu')(T_part_bn_1)
    T_part_dropout_2 = Dropout(0.1)(T_part_cnn_2)
    T_part_maxpooling_2 = MaxPooling1D(2)(T_part_dropout_2)
    T_part_bn_2 = BatchNormalization()(T_part_maxpooling_2)
    T_part_output = Flatten()(T_part_bn_2)

    # 单个心跳的三层卷积
    single_input = Input(shape=single_length)
    single_cnn_1 = Conv1D(nb_filter=32, filter_length=3, strides=2, padding='same',
                          activation='relu')(single_input)
    single_maxpooling_1 = MaxPooling1D(2)(single_cnn_1)
    single_bn_1 = BatchNormalization()(single_maxpooling_1)
    single_cnn_2 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same',
                          activation='relu')(single_bn_1)
    single_dropout_2 = Dropout(0.1)(single_cnn_2)
    single_maxpooling_2 = MaxPooling1D(2)(single_dropout_2)
    single_bn_2 = BatchNormalization()(single_maxpooling_2)
    single_output = Flatten()(single_bn_2)

    all_input = concatenate([P_part_output, QRS_part_output, T_part_output, single_output])

    # 全连接层
    dense_1 = Dense(64, activation='relu')(all_input)
    dropout_1 = Dropout(0.2)(dense_1)
    bn_1 = BatchNormalization()(dropout_1)
    dense_2 = Dense(16, activation='relu')(bn_1)
    dropout_2 = Dropout(0.2)(dense_2)
    bn_3 = BatchNormalization()(dropout_2)
    output = Dense(5, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(bn_3)

    model = Model(inputs=[P_part_input, QRS_part_input, T_part_input, single_input], outputs=output)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    ealry_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=3, verbose=2, mode='auto')

    history = model.fit(x=[data['x_train']['p_wave'], data['x_train']['qrs_wave'], data['x_train']['t_wave'],
                           data['x_train']['single_beat']],
                        y=data['y_train'], batch_size=32, epochs=30, verbose=2,
                        validation_data=([data['x_val']['p_wave'], data['x_val']['qrs_wave'], data['x_val']['t_wave'],
                                          data['x_val']['single_beat']], data['y_val']),
                        shuffle=True, callbacks=[ealry_stop])
    predict_classes = model.predict(
        [data['x_test']['p_wave'], data['x_test']['qrs_wave'], data['x_test']['t_wave'], data['x_test']['single_beat']],
        batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    matrix = matrix.astype('str')
    f = open(write_path, 'a')
    f.write('confusion matrix:' + '\n')
    for i in range(len(matrix)):
        f.write('\t'.join(matrix[i]) + '\n')
    f.write('acc:' + str(acc) + '\n')
    f.close()
    return acc


def experiment_subwaves(read_path, write_path):
    acc_file_path = write_path + 'wave.txt'
    acc_list = []
    data = out_data(read_path=read_path)
    for times in range(10):
        acc = run_model(acc_file_path, data)
        acc_list.append(str(acc))
    f = open(acc_file_path, 'a')
    string = 'acc_300_wave = [' + ','.join(acc_list) + ']' + '\n'
    f.write(string)
    f.close()


if __name__ == '__main__':
    read_path = '/home/lab307/chenbin/data/experiment/beat/'
    write_path = '/home/lab307/chenbin/paper/experiment/classification/wave.txt'
    data_process = out_data(read_path)
    acc_list = []
    # min_delta = [0.002, 0.005, 0.02]
    # for min in min_delta:
    for i in range(10):
        acc = run_model(write_path, data_process, 0.002)
        acc_list.append(str(acc))
    f = open(write_path,'a')
    string = 'acc_300 = [' + ','.join(acc_list) +']' + '\n'
    f.write(string)
    f.close()
