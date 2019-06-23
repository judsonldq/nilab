# -*- encoding:utf-8 -*-

import os
import shutil
import numpy as np


def make_folder(path):
    """
    重新生成某个文件夹
    :param path: str 文件夹路径
    :return: None
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def cut_minutes(label_time, n):
    """
    查找分钟节点
    :param label_time: list 时刻序列
    :param n: int 分钟数
    :return: i: int 对应下标值
    """
    for i in range(len(label_time)):
        if label_time[i] < 60 * n:
            pass
        else:
            return i


def label_transform(label_num, classes):
    """
    标签转换
    :param label_num: int 输入标签
    :param classes: int 标签类别
    :return: 转换后的标签
    """
    if classes == 2:
        classes_dict = {'1': 0, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '10': 1, '11': 1, '12': 11,
                        '13': 1, '31': 1, '34': 1, '37': 1, '38': 1}
    elif classes == 5:
        classes_dict = {'1': 0, '2': 0, '3': 0, '11': 0, '34': 0, '4': 1, '7': 1, '8': 1, '9': 1, '5': 2, '10': 2,
                        '6': 3, '12': 4, '13': 4, '38': 4}
    elif classes == 14:
        classes_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '34': 13, '38': 14}
    elif classes == 15:
        classes_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '10': 8, '11': 9, '12': 10,
                        '13': 11, '31': 12, '34': 13, '37': 14, '38': 15}
    else:
        raise Exception("Invalid input classes!", classes)
    return classes_dict.get(str(int(label_num)))


def add_zero(data, length, mode=None, rate=0, overlapping=0, p_part=0):
    """
    数组补零函数
    :param data: list 待补零数组
    :param length: int 输出长度
    :param mode: string 补零方式
    :param rate: float 偏移比例
    :return: data_process: list 补零之后的数组
    """
    length = int(length)
    if mode == None:
        if len(data) < length:
            data.extend([0 for element in range(length - len(data))])
        return data
    elif mode == 'center':
        offset = int(length * rate)
        real = int(len(data) * rate)
        before = [0 for element in range(offset - real)]
        before.extend(data)
        if len(data) < length:
            before.extend([0 for element in range(length - len(before))])
        return before



def math_feature(data):
    """
    计算数组数学统计特征
    :param data: list 待计算数组
    :return: answer： list 数组统计特征
    """
    if len(data) == 0:
        print("Current data is empty list!")
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    data_postive = [abs(element) for element in data]
    data_mean = round(np.mean(data), 6)
    data_var = round(np.var(data), 6)
    data_postive_mean = round(np.mean(data_postive), 6)
    data_pow_mean = round(np.mean(np.power(data, 2)), 6)
    data_sqrt_mean = round(np.mean(np.sqrt(data_postive)), 6)
    answer = [data_mean, data_var, data_postive_mean, data_pow_mean, data_sqrt_mean]
    return answer


def down_sample(data, down_rate):
    """
    数组下采样
    :param data: list 待下采样数组
    :param down_rate: int 下采样率
    :return: answer: list 下采样数组
    """
    answer = []
    sub_length = len(data) // down_rate
    for i in range(1, sub_length + 1):
        value = round(np.mean(data[down_rate*(i-1):down_rate*i]), 6)
        answer.append(value)
    if sub_length*down_rate < len(data):
        answer.append(round(np.mean(data[sub_length*down_rate:]), 6))
    return answer


def filter_num(length, bias):
    """
    计算神经元个数
    :param length: int 输入数据长度
    :param bias: int  截距
    :return: num: int 神经元个数
    """
    if length == 0:
        print("Input is zero")
        raise ValueError("Invalid input!")
    if length < 129:
        for i in range(1, 8):
            if length <= 2 ** i:
                return 2 ** (i+bias-1)
            else:
                pass
    else:
        for i in range(8, length):
            if length < 2 ** i:
                return 2 **(i+bias-2)
            else:
                pass