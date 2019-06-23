# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Time:     2018/03/09 09:40
@Author:   ChenBin
@Function: constant configuration file
"""

LENGTH = 300
RATE = 0.35
VAL_NUMBER = 5000
KEY_LIST = ['single_beat', 'tri_beat', 'p_wave', 'qrs_wave', 't_wave', "hand_feature"]
CNN_PARAM = {'nb_classes': 5, 'input_size': (128, 1), 'nb_epoch': 10, 'batch_size': 64, 'verbose': 1,
             'kernel_size': {'C1': 3, 'C2': 3, 'C3': 3}, 'nb_num': {'C1': 16, 'C2': 32, 'C3': 64, 'D1': 64},
             'strides': {'C1': 2, 'C2': 2, 'C3': 2}, 'padding': {'C1': 'same', 'C2': 'same', 'C3': 'same'},
             'pool_size': {'C1': 2, 'C2': 2, 'C3': 2}, 'dropout': {'C1': 0.1, 'C2': 0.1, 'C3': 0.2, 'D1': 0.2},
             'optimizer': {'lr': 0.0002, 'rho': 0.9, 'epsilon': 1e-06},
             'early_stop': {'monitor': 'val_loss', 'min_delta': 0.0050, 'patience': 4, 'verbose': 1, 'mode': 'auto'}}

LENGTH_DICT = {'p_para': 0.25, 'qrs_para': 0.25, 't_para': 0.5, 'overlap': 0.06, 'max_length': 300}
