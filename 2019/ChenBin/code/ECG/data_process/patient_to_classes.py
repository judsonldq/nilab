# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time:     2018/03/05 15:30
@Author:   ChenBin
@Function: change heartbeat from patient_based to classes_based
"""


import os
import time
import shutil


def split_class(patient_path, classes_path, classes_num):
    """
    把单个心跳数据从按用户分类文件夹转换为按类别分类文件夹
    :param patient_path: string 用户分类文件路径
    :param classes_path: string 类别分类文件路径
    :param classes_num: int 所有文件类别总数
    :return: None
    """
    folder_list = os.listdir(patient_path)
    if os.path.exists(classes_path):
        shutil.rmtree(classes_path)
    os.mkdir(classes_path)
    for i in range(classes_num):
        os.mkdir(os.path.join(classes_path, str(i)))
    for folder in folder_list:
        folder_path = os.path.join(patient_path, folder)
        file_list = os.listdir(folder_path)
        count = 0
        for files in file_list:
            label = (files.split('.')[0]).split('_')[-1]
            target = (files.split('.')[0]).split('_')[-2]
            new_name = str(count) + '_' + target + '_' +label + '.csv'
            new_path = os.path.join(classes_path, label, new_name)
            old_path = os.path.join(folder_path, files)
            f = open(old_path, 'r')
            g = open(new_path, 'w')
            g.write(f.read())
            f.close()
            g.close()
            count += 1
    print('从用户文件夹到类别文件夹转换完成！')


if __name__ == '__main__':
    patient_path_list = ['/home/lab307/chenbin/data_process/patient/dynamic/20_5/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/20_30/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/20_25/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/24_30/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/24_5/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/24_25/',
                         '/home/lab307/chenbin/data_process/patient/dynamic/44_30/']
    classes_path_list = ['/home/lab307/chenbin/data_process/classes/dynamic/20_5/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/20_30/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/20_25/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/24_30/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/24_5/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/24_25/',
                         '/home/lab307/chenbin/data_process/classes/dynamic/44_30/']
    # patient_path_list = ['/home/chenbin/data_process/patient/paper/44_5/',
    #                      '/home/chenbin/data_process/patient/paper/44_25/']
    # classes_path_list= ['/home/chenbin/data_process/classes/paper/44_5/',
    #                     '/home/chenbin/data_process/classes/paper/44_25/']
    for i in range(len(classes_path_list)):
        patient_path = patient_path_list[i]
        classes_path = classes_path_list[i]
        num = 5
        split_class(patient_path=patient_path,
                    classes_path=classes_path,
                    classes_num=num)
    print("time now:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
