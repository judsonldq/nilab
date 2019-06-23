# -*- encoding:utf-8 -*-

from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, \
Dense, Dropout, BatchNormalization,concatenate
from model.load_data import out_data, compute
import json


def run_model(write_path, data, para_dict):
    single_length = data['x_train']['single_beat'].shape[1]
    p_length = data['x_train']['p_wave'].shape[1]
    qrs_length = data['x_train']['qrs_wave'].shape[1]
    t_length = data['x_train']['t_wave'].shape[1]
    hand_length = data['x_train']['hand_feature'].shape[1]
    P_part_input_shape = (p_length, 1)
    QRS_part_input_shape = (qrs_length, 1)
    T_part_input_shape = (t_length, 1)
    single_input_shape = (single_length, 1)
    hand_selected_input_shape = (hand_length, 1)

    ratio = para_dict['ratio']
    min_delta = para_dict['min_delta']
    bias = para_dict['bias']
    epoch = para_dict['epoch']
    patience = para_dict['patience']

    # P波部分的三层卷积
    P_part_input = Input(shape=P_part_input_shape)
    P_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same', activation='relu')(P_part_input)
    P_part_maxpooling_1 = MaxPooling1D(2)(P_part_cnn_1)
    P_part_bn_1 = BatchNormalization()(P_part_maxpooling_1)
    P_part_dropout_3 = Dropout(0.2)(P_part_bn_1)
    P_part_cnn_3 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same', activation='relu')(P_part_dropout_3)
    P_part_maxpooling_3 = MaxPooling1D(2, padding='same')(P_part_cnn_3)
    P_part_bn_3 = BatchNormalization()(P_part_maxpooling_3)
    P_part_output = Flatten()(P_part_bn_3)

    # QRS波部分的三层卷积
    QRS_part_input = Input(shape=QRS_part_input_shape)
    QRS_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same', activation='relu')(QRS_part_input)
    QRS_part_maxpooling_1 = MaxPooling1D(2)(QRS_part_cnn_1)
    QRS_part_bn_1 = BatchNormalization()(QRS_part_maxpooling_1)
    QRS_part_dropout_3 = Dropout(0.2)(QRS_part_bn_1)
    QRS_part_cnn_3 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same', activation='relu')(QRS_part_dropout_3)

    QRS_part_maxpooling_3 = MaxPooling1D(2, padding='same')(QRS_part_cnn_3)
    QRS_part_bn_3 = BatchNormalization()(QRS_part_maxpooling_3)
    QRS_part_output = Flatten()(QRS_part_bn_3)

    # T波部分的三层卷积
    T_part_input = Input(shape=T_part_input_shape)
    T_part_cnn_1 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same', activation='relu')(T_part_input)
    T_part_maxpooling_1 = MaxPooling1D(2)(T_part_cnn_1)
    T_part_bn_1 = BatchNormalization()(T_part_maxpooling_1)
    T_part_dropout_3 = Dropout(0.2)(T_part_bn_1)
    T_part_cnn_3 = Conv1D(nb_filter=8, filter_length=3, strides=2, padding='same', activation='relu')(T_part_dropout_3)
    T_part_maxpooling_3 = MaxPooling1D(2, padding='same')(T_part_cnn_3)
    T_part_bn_3 = BatchNormalization()(T_part_maxpooling_3)
    T_part_output = Flatten()(T_part_bn_3)

    # 单个心跳的三层卷积
    single_input = Input(shape=single_input_shape)
    single_cnn_1 = Conv1D(nb_filter=32, filter_length=3, strides=2, padding='same', activation='relu')(single_input)
    single_maxpooling_1 = MaxPooling1D(2)(single_cnn_1)
    single_bn_1 = BatchNormalization()(single_maxpooling_1)
    single_dropout_3 = Dropout(0.2)(single_bn_1)
    single_cnn_3 = Conv1D(nb_filter=16, filter_length=3, strides=2, padding='same', activation='relu')(single_dropout_3)
    single_maxpooling_3 = MaxPooling1D(2)(single_cnn_3)
    single_bn_3 = BatchNormalization()(single_maxpooling_3)
    single_output = Flatten()(single_bn_3)

    hand_input = Input(shape=hand_selected_input_shape)
    dense_hand_1 = Dense(8, activation='relu')(hand_input)
    bn_hand_1 = BatchNormalization()(dense_hand_1)
    hand_output = Flatten()(bn_hand_1)

    all_input = concatenate([single_output, P_part_output, QRS_part_output, T_part_output, hand_output])

    # 全连接层
    dense_1 = Dense(64, activation='relu')(all_input)
    bn_1 = BatchNormalization()(dense_1)
    dropout_1 = Dropout(0.2)(bn_1)
    dense_3 = Dense(16, activation='relu')(dropout_1)
    bn_3 = BatchNormalization()(dense_3)
    output = Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(bn_3)

    model = Model(inputs=[single_input, P_part_input, QRS_part_input, T_part_input, hand_input],
                  outputs=output)

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=2, mode='auto')

    history = model.fit(x=[data['x_train']['single_beat'], data['x_train']['p_wave'], data['x_train']['qrs_wave'],
                           data['x_train']['t_wave'], data['x_train']['hand_feature']],
                        y=data['y_train'], batch_size=32, epochs=epoch, verbose=2,
                        validation_data=([data['x_val']['single_beat'], data['x_val']['p_wave'], data['x_val']['qrs_wave'],
                                          data['x_val']['t_wave'], data['x_val']['hand_feature']], data['y_val']),
                        shuffle=True, callbacks=[early_stopping])
    predict_classes = model.predict(
        [data['x_test']['single_beat'], data['x_test']['p_wave'], data['x_test']['qrs_wave'], data['x_test']['t_wave'],
        data['x_test']['hand_feature']],
        batch_size=32, verbose=2)
    matrix, acc = compute(data['y_test'], predict_classes)
    matrix = matrix.astype('str')
    f = open(write_path, 'a')
    f.write('confusion matrix:' + '\n')
    for i in range(len(matrix)):
        f.write('\t'.join(matrix[i]) + '\n')
    f.write('acc:' + str(acc) + '\n')
    f.close()
    return acc, model


def experiment_hand_feature(para_dict):
    read_path = '../beat/'
    acc_file_path = './hand_no_tri.txt'
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
# def triple():
    para_dict = {'min_delta': 0.002, 'ratio': 2, 'bias': 0, 'epoch': 50, 'times': 10, 'patience': 4}
    min_delta_list = [0.002, 0.005, 0.02]
    for i in range(len(min_delta_list)):
        para_dict['min_delta'] = min_delta_list[i]
        experiment_hand_feature(para_dict)