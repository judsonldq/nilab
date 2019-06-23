# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time:     2018/03/08 20:15
@Author:   ChenBin
@Function: Check the length of files lines
"""

import os
import numpy as np


def check_lines(path, lines, mode='file'):
    """
    检测文件夹或子文件夹下的文件行数是否等于lines
    :param path: string 待检查的文件路径
    :param lines: int 指定的文件行数
    :param mode: string 文件夹下是文件还是文件夹
    :return: None
    """
    if mode == 'file':
        count_lines(path=path, lines=lines)
    elif mode == 'folder':
        folder_list = os.listdir(path)
        for folder in folder_list:
            folder_path = os.path.join(path, folder)
            count_lines(path=folder_path, lines=lines)


def count_lines(path, lines):
    """
    检测文件行数是否等于lines
    :param path: string 待检查的文件路径
    :param lines: int 指定的文件行数
    :return: None
    """
    num = 0
    file_list = os.listdir(path)
    for files in file_list:
        file_path = os.path.join(path, files)
        count = len(open(file_path, 'r').readlines())
        if count == lines:
            pass
        else:
            num += 1
            print('文件长度不符合要求！')
            print(file_path, '，长度为：' + str(count))
    print(num)


def check_label_from_file(path):
    """
    统计文件夹中各类数目
    :param path: string 待统计的样本路径
    :return: class_dict: dict 统计出的各类别的数量
    """
    class_dict = {}
    file_list = os.listdir(path)
    for files in file_list:
        label = int(files.split('.')[0].split('_')[-1])
        if str(label) in class_dict.keys():
            class_dict[str(label)] += 1
        else:
            class_dict[str(label)] = 1
    return class_dict


def check_label_from_list(data):
    """
    统计列表中各类数目
    :param data: list 待统计类别数目的列表
    :return: class_dict:dict 统计出的各类别的数量
    """
    class_dict = {}
    if isinstance(data[0], np.ndarray):
        for array in data:
            array = list(array)
            label = array.index(max(array))
            if str(label) in class_dict.keys():
                class_dict[str(label)] += 1
            else:
                class_dict[str(label)] = 1
    else:
        for label in data:
            label = int(label)
            if str(label) in class_dict.keys():
                class_dict[str(label)] += 1
            else:
                class_dict[str(label)] = 1
    return class_dict

# """
if __name__ == '__main__':
    path = '/home/chenbin/data_process/classes/paper/44_5/4/'
    print(check_label_from_file(path))

# """[14543,275,1100,180,0]
