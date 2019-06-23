# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time:     2018/03/08 17:20
@Author:   ChenBin
@Function: a example of use class CNN
"""

from model.conv import CNN
from data_process.load_data import load_data
from conf.config import CNN_PARAM
from data_process.check import check_lines, check_label_from_file, check_label_from_list


def cnn_model():
    train_path = '/home/chenbin/data_process/experiment/5/train/'
    test_path = '/home/chenbin/data_process/experiment/5/test/'
    val_path = '/home/chenbin/data_process/experiment/5/val/'
    x_train, y_train = load_data(train_path, 5)
    x_val, y_val = load_data(val_path, 5)
    x_test, y_test = load_data(test_path, 5, 'test')
    print(check_label_from_list(y_train))
    print(check_label_from_list(y_val))
    print(check_label_from_list(y_test))
    cnn = CNN()
    cnn.set_params(params=CNN_PARAM)
    history, model = cnn.fit(x_train=x_train, y_train=y_train,
                             x_val=x_val, y_val=y_val)
    y_pred = cnn.predict(x_test=x_test, model=model)
    conf_mat = cnn.confusion_matrix(y_true=y_test, y_predict=y_pred)
    print(conf_mat)
    print(cnn.classification_report(y_true=y_test, y_predict=y_pred))
    cnn.save_model(model, '/home/chenbin/data_process/experiment/test/')
    cnn.draw_history(history=history)
    cnn.draw_heat_map(conf_mat)


if __name__ == '__main__':
    cnn_model()

