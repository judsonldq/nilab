# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time:     2018/03/04 19:20
@Author:   ChenBin
@Function: add Gaussian noise to data_process
"""
import random
import numpy as np
import os


def add(data, scale):
    """
    给一列数据加上高斯噪声
    :param data: list 原始数据
    :param scale: float 高斯方差是数据方差的倍数
    :return:
    """
    # mean = np.mean(data_process)
    variable = np.var(data)
    for i in range(len(data)-5):
        data[i] = data[i] + round(random.gauss(0, scale*variable))
    return data


def upsample(path, scale, up_list):
    file_list = os.listdir(path)
    for files in file_list:
        label = int(files.split('.')[0].split('_')[-1])
        if up_list[label] > 0:
            for j in range(up_list[label]):
                data = []
                f = open(os.path.join(path,files),'r')
                g = open(os.path.join(path,'guass_'+str(j)+files),'a')
                for lines in f.readlines():
                    data.append(float(lines.strip()))
                data = add(data, scale)
                for i in range(len(data)):
                    g.write(str(data[i])+'\n')
                f.close()
                g.close()


if __name__ == '__main__':
    path = '/home/chenbin/data_process/experiment/dynamic/train/'
    up_list = [0,2,1,3,5]
    upsample(path,0.05,up_list)
