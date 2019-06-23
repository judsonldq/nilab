# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time:     2018/03/04 19:20
@Author:   ChenBin
@Function: split Continuous heartbeat signals(ecg) to a single heartbeat signal
"""

import os
import shutil
import numpy as np


def load_data(file_path, mode):
    """
    读取心跳数据和标签数据
    :param file_path: string 待读取文件的路径
    :param mode: int 标签数据读取模式，是读取全部还是某一部分，可选参数0,1,2
                     30：读取前半个小时
                     5：读取前5分钟
                     25：读取5-30分钟
    :return: data_process: list 读取的数据
             time: list 读取该数据的时间
    """
    fr = open(file_path, 'r')
    time = []
    data = []
    for lines in fr.readlines():
        temp = lines.strip().split(',')
        temp[0] = float(temp[0])
        temp[1] = float(temp[1])
        # 读取前半个小时的数据(所有)
        if mode == '30':
            time.append(temp[0])
            data.append(temp[1])
            # if temp[0] < 1800.0:
            #     time.append(temp[0])
            #     data_process.append(temp[1])
            #     continue
            # else:
            #     break
        # 读取前5分钟的数据
        elif mode == '5':
            if temp[0] < 300.0:
                time.append(temp[0])
                data.append(temp[1])
                continue
            else:
                break
        # 读取5-30分钟的数据
        elif mode == '25':
            if 300.0 <= temp[0]:
                time.append(temp[0])
                data.append(temp[1])
            else:
                continue
        else:
            raise Exception
            # time.append(temp[0])
            # data_process.append(temp[1])
    print('数据读取完成！')
    return data, time


def find_Rtime(label_time, data_time):
    """
    获得R波尖峰所在时间轴对应下标位置
    :param label_time: list R波尖峰标签时间的列表
    :param data_time: list 心跳数据记录点时间的列表
    :return: Rtime: list R波尖峰所对应的心跳数据时间下标的列表
    """
    j = 0
    Rtime = []
    for i in range(len(data_time)-1):
        if j < len(label_time):
            if data_time[i] <= label_time[j] < data_time[i+1]:
                Rtime.append(i)
                j += 1
        else:
            break
    return Rtime


"""
def chip_beat(Rtime, data_list, beat_num, rate, beat_length):
    \"""
    把连续的不定长的心跳切分成单个的定长心跳
    :param Rtime: list R波尖峰对应时间轴下标，用来标志每一个心跳原点
    :param data_list: list 连续心跳数据
    :param beat_num: int 用来计算单个心拍平均长度的总心拍数, 最好为偶数
    :param rate: float R波前点数占总点数比率
    :param beat_length: int 定长心跳的长度
    :return: all_beat_list: list of pairs 单个心跳list的集合
    \"""
    all_beat_list = []
    length = len(data_list)
    if beat_num % 2 == 0:
        beat_before = beat_after = int(beat_num / 2)
    else:
        beat_before = int(beat_num / 2)
        beat_after = beat_before + 1
    for i in range(len(Rtime)):
        if i < beat_before:
            single_length = int((Rtime[beat_num] - Rtime[0])/beat_num)
        elif i > beat_after:
            single_length = int((Rtime[-beat_num-1] - Rtime[-1])/beat_num)
        else:
            single_length = int((Rtime[i+beat_after] - Rtime[i-beat_before])/beat_num)
        before = int(single_length*rate)
        after = single_length - before
        if Rtime[i] < before:
            beat = data_list[:before+after]
        elif length - Rtime[i] > after:
            beat = data_list[-before-after:-1]
        else:
            beat = data_list[Rtime[i]-before:Rtime[i]+after]
        single_beat = trans(num_list=beat, length=beat_length)
        all_beat_list.append(single_beat)
    print('心跳切分完成！')
    return all_beat_list
"""



"""
def trans(num_list, length):
    把一个动长心跳转换为定长心跳
    :param num_list: list 待转换的动长心跳
    :param length: int 定长的长度
    :return: beat_list
    beat_list = []
    interval_list = np.arange(0, len(num_list), (len(num_list)/(length*1.0))).tolist()
    for i in range(len(interval_list)-1):
        low = int(np.ceil(interval_list[i]))
        up = int(np.floor(interval_list[i + 1]))
        if up >= low:
            beat_list.append(round(sum(num_list[low:up+1])/(len(num_list[low:up+1])*1.0), 5))
        else:
            print('\033[1;31;40m%s' % '该心跳原长度小于转换后的心跳长度！')
            print('\033[0m')
            print('心跳原长度：', len(num_list))
            if len(beat_list) != 0:
                if len(num_list) > low:
                    beat_list.append(round((beat_list[-1] + num_list[low])/2.0, 5))
                else:
                    beat_list.append(round((beat_list[-1] + num_list[up]) / 2.0, 5))
            else:
                beat_list.append(num_list[0])
    return beat_list


"""

"""
def trans(num_list, length):   #输入经过切分处理的周期序列num_list和期望输出数据个数NUM
    index_to_find = np.arange(0, len(num_list) - 1, len(num_list) / float(length)).tolist()
    result = []
    for i in index_to_find:
        if i % 1 == 0:
            result.append(num_list[int(i)])
        else:
            i_up = int(np.ceil(i))
            i_down = int(np.floor(i))
            result.append(round(num_list[i_up]*(i-i_down)+num_list[i_down]*(i_up-i), 5))
    return result
"""


def chip_beat(Rtime, data_list, before, after):
    """
    把连续的心跳切分成单个的心跳
    :param Rtime: list R波尖峰对应时间轴下标，用来标志每一个心跳原点
    :param data_list: list 连续心跳数据
    :param before: int R波尖峰之前取点数
    :param after: int R波尖峰之后取点数
    :return: all_beat_list: list of pairs 单个心跳list的集合
    """
    all_beat_list = []
    length = len(data_list)
    for i in range(1, len(Rtime)-1):
        # 对Rtime标志位判断
        if before < Rtime[i] < length - after:
            single_beat = data_list[Rtime[i]-before:Rtime[i]+after]
        elif Rtime[i] < before:
            single_beat = data_list[0:before+after]
            print('the beginning is less than %s' % before)
        else:
            single_beat = data_list[-(before+after)-1:-1]
            print('the ending is less than %s' % after)
        #添加RR间隔
        last_RR = Rtime[i] - Rtime[i-1]
        next_RR = Rtime[i+1] - Rtime[i]
        compare_ratio = round(last_RR/(next_RR*1.0), 5)
        sum_ratio = round(abs(last_RR-next_RR)/(1.0*(last_RR+next_RR)), 5)
        first_ratio = round(last_RR/(1.0*(last_RR+next_RR)), 5)
        single_beat.extend([last_RR, next_RR, compare_ratio, sum_ratio, first_ratio])
        all_beat_list.append(single_beat)
    print('心跳切分完成！')
    return all_beat_list


# def detect_peak(data_process):
#     max_data = max(data_process)
#     R_index = data_process.index(max_data)
#     return Q_index,R_index,S_index


def write_beat(all_beat_list, labels, path, k, num):
    """
    把切分好的单个心跳写入文件
    :param all_beat_list: list 切分好的单个心跳列表
    :param labels: list 每个心跳对应的标签
    :param path: string 写入文件的路径
    :param k: int 该心跳所属病人编号
    :param num: int 写入文件时心跳分类数目 目前可用2,5,14,15分类
    :return: None
    """
    m = len(all_beat_list)
    file_list = os.listdir(path)
    if str(k) not in file_list:
        folder_name = os.path.join(path, str(k))
        os.mkdir(folder_name)
    folder_path = os.path.join(path, str(k))
    if num == 2:
        classes_dict = {'1': 0, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '10': 1, '11': 1, '12': 11,
                        '13': 1, '31': 1, '34': 1, '37': 1, '38': 1}
    elif num == 5:
        classes_dict = {'1': 0, '2': 0, '3': 0, '11': 0, '34': 0, '4': 1, '7': 1, '8': 1, '9': 1, '5': 2, '10': 2,
                        '6': 3, '12': 4, '13': 4, '38': 4}
    elif num == 14:
        classes_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '34': 13, '38': 14}
    elif num == 15:
        classes_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '10': 8, '11': 9, '12': 10,
                        '13': 11, '31': 12, '34': 13, '37': 14, '38': 15}
    for i in range(0, m):
        if str(int(labels[i])) in classes_dict.keys():
            half_write_name = str(k) + '_' + str(i) + '_' + str(classes_dict[str(int(labels[i]))]) + '.csv'
            write_name = os.path.join(folder_path, half_write_name)
            f = open(write_name, 'a')
            write_data = [str(element) for element in all_beat_list[i]]
            f.write(','.join(write_data) + '\n')
            f.close()
    print('所有数据已经写入文件！')


# def chip(read_path, write_path, num, rate, length, class_num, patient_num='44', read_mode='30'):
#     """
#     把所有连续心跳切分成单个心跳，并把标签转换为指定类型标签
#     :param read_path: string 读取原始数据的路径
#     :param write_path: string 写入新数据的路径
#     :param num: int 计算平均心跳长度
#     :param rate: float R波尖峰前的取点数占比
#     :param length: int 输出心跳长度
#     :param class_num: string 新分类的类别数目
#     :param patient_num: string 病人序号列表 可选 '44', '24', '20' 默认'44'
#     :param read_mode: int 读取模式 可选'30', '25', '5' 默认'30'
#     :return: None
#     """
def chip(read_path, write_path, before, after, class_num, patient_num='44', read_mode='30'):
    """
    把所有连续心跳切分成单个心跳，并把标签转换为指定类型标签
    :param read_path: string 读取原始数据的路径
    :param write_path: string 写入新数据的路径
    :param before: int 在R波尖峰之前取的点数
    :param after: int 在R波尖峰之后取的点数
    :param class_num: string 新分类的类别数目
    :param patient_num: string 病人序号列表 可选 '44', '24', '20' 默认'44'
    :param read_mode: int 读取模式 可选'30', '25', '5' 默认'30'
    :return: None
    """
    if patient_num == '44':
        patient = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124,
                   200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 230,
                   231, 232, 233, 234]
    elif patient_num == '20':
        patient = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124]
    elif patient_num == '24':
        patient = [200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 219, 220, 221, 222, 223, 228, 230,
                   231, 232, 233, 234]
    data_read_path = os.path.join(read_path, 'read_data/')
    label_read_path = os.path.join(read_path, 'read_atr/')
    for k in patient:
        # print(str(k) + 'is start.', write_path,read_mode)
        # try:
        data, data_time = load_data(file_path=data_read_path + str(k) + 'data_process.csv',
                                    mode=read_mode)
        print('读取心跳数据：', len(data), len(data_time))
        labels, labels_time = load_data(file_path=label_read_path + str(k) + 'label.csv',
                                        mode=read_mode)
        print('读取心跳标签数据：', len(labels), len(labels_time))
        RTime = find_Rtime(label_time=labels_time,
                           data_time=data_time)
        print('获得R波尖峰：', len(RTime))
        all_beat_list = chip_beat(Rtime=RTime,
                                  data_list=data,
                                  before=before,
                                  after=after)
        # all_beat_list = chip_beat(Rtime=RTime,
        #                           data_list=data_process,
        #                           beat_num=num,
        #                           rate=rate,
        #                           beat_length=length)
        print('获得切分数组：', len(all_beat_list), len(all_beat_list[0]))
        write_beat(all_beat_list=all_beat_list,
                   labels=labels,
                   path=write_path,
                   k=k,
                   num=class_num)
        print(str(k) + ' is done!')
        # except Exception as e:
        #     print('There is a bug in ' + str(k) + '!')
        #     print(e)


if __name__ == '__main__':
    write_path_list = ['/home/chenbin/data_process/patient/200/20_5/',
                       '/home/chenbin/data_process/patient/200/20_30/',
                       '/home/chenbin/data_process/patient/200/20_25/',
                       '/home/chenbin/data_process/patient/200/24_30/',
                       '/home/chenbin/data_process/patient/200/24_5/',
                       '/home/chenbin/data_process/patient/200/24_25/',
                       '/home/chenbin/data_process/patient/200/44_30/']
    # write_path_list = ['/home/chenbin/data_process/patient/paper/44_5/',
    #                    '/home/chenbin/data_process/patient/paper/44_25/']
    for write_path in write_path_list:
        mode_list = write_path.split('/')[-2].split('_')
        read_mode = mode_list[-1]
        patient_num = mode_list[0]
        read_path = '/home/chenbin/data_process/MITDB/'
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        os.mkdir(write_path)
        class_num = 5
        # 常量赋值
        # num = 2
        # rate = 0.2
        # length = 128
        # chip(read_path=read_path,
        #      write_path=write_path,
        #      num=num,
        #      rate=rate,
        #      length=length,
        #      class_num=class_num,
        #      patient_num=patient_num,
        #      read_mode=read_mode)
        before = 25
        after = 275
        chip(read_path=read_path,
             write_path=write_path,
             before=before,
             after=after,
             class_num=class_num,
             patient_num=patient_num,
             read_mode=read_mode)

