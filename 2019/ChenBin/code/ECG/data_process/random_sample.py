# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import shutil
import random
import numpy as np

def train_compose(read_path, write_path, sample_class):
    folder_list = os.listdir(read_path)
    for folder in folder_list:
        folder_path = os.path.join(read_path, folder)
        file_list = os.listdir(folder_path)
        train_set = random.sample(file_list, sample_class[int(folder)])
        test_set = [element for element in file_list if element not in train_set]
        train_write_path = os.path.join(write_path, 'train')
        test_write_path = os.path.join(write_path, 'test')
        write_file(train_set, read_path, train_write_path)
        write_file(test_set, read_path, test_write_path)
    print('训练集生成完成！')



def write_file(file_list, read_path, write_path):
    """
    write files
    :param file_list: list list of files name
    :param read_path: string folder path of files to read
    :param write_path: string folder path of files to read
    :return: None
    """
    for files in file_list:
        file_read_path = os.path.join(read_path, files)
        file_write_path = os.path.join(write_path, files)
        f = open(file_read_path, 'r')
        g = open(file_write_path, 'w')
        g.write(f.read())
        f.close()
        g.close()
