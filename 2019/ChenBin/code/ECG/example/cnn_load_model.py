# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time:     2018/03/13 09:20
@Author:   ChenBin
@Function: a example of use class CNN by loading models
"""

from model.conv import CNN
from data_process.load_data import load_data
from conf.config import CNN_PARAM
from data_process.check import check_lines, check_label_from_file, check_label_from_list

def cnn_model(test_path):
    x_test, y_test = load_data(test_path, 5, 'test')
    print(check_label_from_list(y_test))
    cnn = CNN()
    cnn.set_params(params=CNN_PARAM)
    model = cnn.load_model(load_path='/home/chenbin/data_process/experiment/test/', version=0)
    print(type(model))
    y_pred = cnn.predict(x_test=x_test, model=model)
    conf_mat = cnn.confusion_matrix(y_true=y_test, y_predict=y_pred)
    print(conf_mat)
    print(cnn.classification_report(y_true=y_test, y_predict=y_pred))
    cnn.draw_heat_map(conf_mat)


if __name__ == '__main__':
    test_path = '/home/chenbin/data_process/patient/5/24_30/200/'
    cnn_model(test_path=test_path)
