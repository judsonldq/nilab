# !/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import regularizers
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from data.load_data import out_data, compute


def run_model(data):
    input_ecg = Input(data['x_train'].shape[1:])

    # encode_0 = BatchNormalization()(input_ecg)
    encode_1 = Conv1D(nb_filter=512, filter_length=3, strides=2, padding='same', activation='relu'
                      )(input_ecg)
    encode_2 = MaxPooling1D(4, padding='same')(encode_1)
    encode_3 = BatchNormalization()(encode_2)
    encode_4 = Conv1D(nb_filter=256, filter_length=3, strides=2, padding='same', activation='relu'
                      )(encode_3)
    dropout_1 = Dropout(0.1)(encode_4)
    encode_5 = MaxPooling1D(4, padding='same')(dropout_1)
    encode_6 = BatchNormalization()(encode_5)
    encode_7 = Conv1D(nb_filter=128, filter_length=3, strides=2, padding='same', activation='relu'
                      )(encode_6)
    dropout_2 = Dropout(0.2)(encode_7)
    encode_8 = MaxPooling1D(4, padding='same')(dropout_2)
    encode_9 = BatchNormalization()(encode_8)
    encode_10 = Flatten()(encode_9)

    dense = Dense(512, activation='relu')(encode_10)
    dropout = Dropout(0.2)(dense)
    output = Dense(252, activation='softmax', kernel_regularizer=regularizers.l2(0.00))(dropout)

    model = Model(input_ecg, output)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    EarlyStopping(monitor='val_loss', min_delta=0.0050, patience=3, verbose=2, mode='auto')

    history = model.fit(x=data['x_train'], y=data['y_train'], batch_size=32, epochs=15,
                        verbose=2, validation_data=(data['x_val'], data['y_val']),
                        shuffle=True)
    predict_classes = model.predict(data['x_test'], batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    matrix = matrix.astype('str')
    # f = open(write_path, 'a')
    # f.write('confusion matrix:' + '\n')
    # for i in range(len(matrix)):
    #     f.write('\t'.join(matrix[i]) + '\n')
    # f.write('acc:' + str(acc) + '\n')
    # f.close()
    # return acc


def experiment_find_offset(read_path):
    acc_list = []
    data = out_data(read_path=read_path)
    for times in range(10):
        acc = run_model(data)
        acc_list.append(str(acc))
    # f = open(acc_file_path, 'a')
    # string = 'acc_300_' + offset + ' = [' + ','.join(acc_list) + ']' + '\n'
    # f.write(string)
    # f.close()



if __name__ == '__main__':
    read_path = '/home/lab307/chenbin/data/experiment/identification/'
    experiment_find_offset(read_path)