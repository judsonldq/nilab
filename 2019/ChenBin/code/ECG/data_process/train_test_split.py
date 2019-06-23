# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time:     2018/03/05 20:00
@Author:   ChenBin
@Function: change heartbeat from classes_based to train_test_split
"""

import os
import time
import shutil
import random


def split_train_test(data_path, experiment_path, test_mode='all', val_num=5000):
    """
    生成实验所需的训练集，验证集和测试集
    :param data_path: string 实验数据源文件路径
    :param experiment_path: string 生成实验数据保存文件路径
    :param test_mode: string 测试集大小 'half':选用前20个人的后25分钟作为测试集  'all':选用44个人的数据作为测试集 默认'all'
    :param val_num: int 验证集的数目
    :return: None
    """
    if os.path.exists(experiment_path):
        shutil.rmtree(experiment_path)
    os.mkdir(experiment_path)
    train_save_path = experiment_path + 'train/'
    os.mkdir(train_save_path)
    test_save_path = experiment_path + 'test/'
    os.mkdir(test_save_path)
    val_save_path = experiment_path + 'val/'
    os.mkdir(val_save_path)
    # train_data_path = data_path + '20_5/'
    # test_data_path_1 = data_path + '20_25/'
    # test_data_path_2 = data_path + '24_30/'
    train_data_path = data_path + '24_5/'
    train_sample_path = data_path + '20_30/'
    test_data_path_1 = data_path + '24_25/'
    test_data_path_2 = data_path + '20_30/'
    val_data_path = test_save_path
    # 生成训练集
    get_train_set(sample_path=train_sample_path,
                  data_path=train_data_path,
                  save_path=train_save_path)
    # 生成测试集
    get_test_set(data_path_1=test_data_path_1,
                 data_path_2=test_data_path_2,
                 save_path=test_save_path,
                 mode=test_mode)
    # 生成验证集
    if val_num > 0:
        get_val_set(data_path=val_data_path,
                    save_path=val_save_path,
                    num=val_num)
    print('完成实验各项数据集生成！')


def get_train_set(sample_path, data_path, save_path):
    """
    生成训练数据集
    :param data_path: string 前20个病人的前5分钟的样本的路径
    :param sample_path: string 前20个病人的前30分钟的样本的路径,用于全局抽样
    :param save_path: string 训练数据集的保存路径
    :return: None
    """
    # 取样前20个病人的前30分钟的245个样本
    random_sample(data_path=sample_path,
                  save_path=save_path)
    count = 245
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        folder_path = os.path.join(data_path, folder)
        file_list = os.listdir(folder_path)
        for files in file_list:
            read_path = os.path.join(folder_path, files)
            new_name = str(count) + '_' + files.split('_')[-1]
            write_path = os.path.join(save_path, new_name)
            count += 1
            f = open(read_path, 'r')
            g = open(write_path, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()
    print('完成训练数据集生成！')


def random_sample(data_path, save_path):
    """
    从前20个病人的前5分钟中筛选一些样本[75, 75, 75 ,13, 7]
    :param data_path: string 前20个病人的30分钟的样本的路径
    :param save_path: string 筛选的样本的存储路径
    :return: None
    """
    folder_list = os.listdir(data_path)
    classes_sample_dict = {'0': 75, '1': 75, '2': 75, '3': 13, '4': 7}
    classes_dict = {}
    count = 0
    for folder in folder_list:
        folder_path = os.path.join(data_path, folder)
        classes_dict[folder] = os.listdir(folder_path)
    for k in classes_dict.keys():
        sample_list = random.sample(classes_dict[k], classes_sample_dict[k])
        for j in range(len(sample_list)):
            read_path = os.path.join(data_path, k, sample_list[j])
            new_name = str(count) + '_' + sample_list[j].split('_')[-1]
            write_path = os.path.join(save_path, new_name)
            count += 1
            f = open(read_path, 'r')
            g = open(write_path, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()
    print('完成训练集全局采样！')


def get_test_set(data_path_1, data_path_2, save_path, mode):
    """
    生成测试数据集
    :param data_path_1: string 前20个人的后25分钟数据的文件路径
    :param data_path_2: string 后24个人的30分钟数据的文件路径
    :param save_path: string 合成测试集文件保存路径
    :param mode: string 是选择只生成前20个人的数据做测试集还是44个人的数据做测试集,可选'all':44 'half':20
    :return: None
    """
    count = write_file(data_path=data_path_1, save_path=save_path, count=0)
    if mode == 'all':
        count = write_file(data_path=data_path_2, save_path=save_path, count=count)
    print('完成测试集生成！')


def write_file(data_path, save_path, count):
    """
    批量把文件从按类别分类的子文件夹写入一个文件夹下
    :param data_path: string 文件源路径
    :param save_path: string 文件输出路径
    :param count: int 文件名的序号
    :return: count int 文件名的序号
    """
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        folder_path = os.path.join(data_path, folder)
        file_list = os.listdir(folder_path)
        for files in file_list:
            name = str(count) + '_' + files.split('_')[-1]
            count += 1
            old_path = os.path.join(folder_path, files)
            new_path = os.path.join(save_path, name)
            f = open(old_path, 'r')
            g = open(new_path, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()
    return count


def get_val_set(data_path, save_path, num):
    """
    生成验证数据集
    :param data_path: string 验证数据集源的文件路径
    :param save_path: string 验证数据集保存路径
    :param num: int 验证数据集的数目
    :return: None
    """
    file_list = os.listdir(data_path)
    val_list = random.sample(file_list, num)
    for files in val_list:
        old_path = os.path.join(data_path, files)
        new_path = os.path.join(save_path, files)
        os.rename(old_path, new_path)
    print('完成验证集生成！')


if __name__ == '__main__':
    data_path = '/home/lab307/chenbin/data_process/classes/dynamic/'
    experiment_path = '/home/lab307/chenbin/data_process/experiment/dynamic/'
    test_mode = 'all'
    val_num = 5000
    split_train_test(data_path=data_path,
                     experiment_path=experiment_path,
                     test_mode=test_mode,
                     val_num=5000)
    print("time now:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
