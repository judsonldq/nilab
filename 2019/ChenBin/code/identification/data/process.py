# -*- encoding:utf-8 -*-
import os
import pywt
import json
import random
import numpy as np
from tool.utils import make_folder


def data_process(read_path, write_path, para_dict):
    """
    数据处理函数
    :param read_path: string 源数据路径
    :param write_path: string 处理之后的数据路径
    :return: None
    """
    # SAMPLE_SECONDS = 8
    # TRAIN_SECONDS = 90
    # TEST_SECONDS = 90
    # TRAIN_SAMPLES = 200
    # TEST_SAMPLES = 200
    # VAL_SAMPLES = 100
    SAMPLE_SECONDS = para_dict['sample_seconds']
    TRAIN_SECONDS = para_dict['train_seconds']
    TEST_SECONDS = para_dict['test_seconds']
    TRAIN_SAMPLES = para_dict['train_samples']
    TEST_SAMPLES = para_dict['test_samples']
    VAL_SAMPLES = para_dict['val_samples']
    train_num = val_num = test_num = 0
    train_path = write_path + 'train/'
    test_path = write_path + 'test/'
    val_path = write_path + 'val/'
    print("rebuild folders...")
    make_folder(train_path)
    make_folder(val_path)
    make_folder(test_path)
    print("generate samples...")
    file_list = os.listdir(read_path)

    # 产生label
    file_list.sort()
    tag_dict = {}
    for i in range(len(file_list)):
        tag_dict[file_list[i]] = i
    for file in file_list:
        # tag = file.split('.')[0]
        tag = tag_dict[file]
        file_path = os.path.join(read_path, file)
        data = read_files(file_path, mode='single')
        # print(type(data))
        # print("data:",data)
        if VAL_SAMPLES > 0:
            train, val, test = generate_samples(data, TRAIN_SECONDS, TEST_SECONDS, TRAIN_SAMPLES,
                                                TEST_SAMPLES, SAMPLE_SECONDS, VAL_SAMPLES)
            val_num = write_files(val, val_path, tag, val_num)
        else:
            train, test = generate_samples(data, TRAIN_SECONDS, TEST_SECONDS, TRAIN_SAMPLES,
                                           TEST_SAMPLES, SAMPLE_SECONDS)
        train_num = write_files(train, train_path, tag, train_num)
        test_num = write_files(test, test_path, tag, test_num)
    print("samples have generated!")


def read_files(read_path, mode='all'):
    """
    读取数据
    :param read_path: string 源数据路径
    :return: data: list 读取到的数据
    """
    if mode == 'all':
        with open(read_path, 'r') as f:
            data = []
            for line in f.readlines():
                temp = [round(float(element), 6) for element in line.strip().split(',')]
                data.append(temp)
        return data
    elif mode == 'single':
        with open(read_path, 'r') as f:
            data = []
            for line in f.readlines():
                temp = float(line.strip().split(',')[1])
                data.append(temp)
        return data
    else:
        print("The mode of function is wrong!")
        raise TypeError


def generate_samples(data, train_seconds, test_seconds, train_samples, test_samples, sample_second=2,
                     val_samples=0, down_rate=1, level=5):
    """
    生成数据集
    :param data: list 源数据
    :param train_seconds: int train数据总时长
    :param train_samples: int train样本总数
    :param test_samples: int test样本总数
    :param sample_second: int 单样本时长
    :param val_samples: int val样本总数
    :param down_rate: int 下采样率
    :param level: int 小波变换阶数
    :return: train,val,test list 数据集
    """
    frequency = 360
    bias = frequency * 300
    sample_length = frequency * sample_second
    train_start_index = bias
    train_end_index = bias + train_seconds*frequency - sample_length
    test_start_index = bias + train_seconds*frequency
    test_end_index = bias + (train_seconds + test_seconds)*frequency - sample_length
    train_samples = get_samples(data=data,
                                length=sample_length,
                                sample_num=train_samples,
                                sample_start=train_start_index,
                                sample_end=train_end_index,
                                down_rate=down_rate,
                                level=level)
    test_samples = get_samples(data=data,
                               length=sample_length,
                               sample_num=test_samples,
                               sample_start=test_start_index,
                               sample_end=test_end_index,
                               down_rate=down_rate,
                               level=level)
    # 对时间序列进行小波变换
    if val_samples > 0:
        val_samples = get_samples(data=data,
                                  length=sample_length,
                                  sample_num=val_samples,
                                  sample_start=test_start_index,
                                  sample_end=test_end_index,
                                  down_rate=down_rate,
                                  level=level)
        return train_samples, val_samples, test_samples
    else:
        return train_samples, test_samples


def get_samples(data, length, sample_num, sample_start, sample_end, down_rate, level):
    """
    样本切分
    :param data: np.array 源数据
    :param length: int 样本长度
    :param sample_num: int 样本数
    :param sample_start: int 源数据起始点
    :param sample_end: int 源数据结束点
    :param down_rate: int 下采样率
    :param level: int 小波变换阶数
    :return: samples list 样本
    """
    samples = []
    channel_num = len(np.shape(data))
    if channel_num == 1:
        for i in range(sample_num):
            start = random.randint(sample_start, sample_end)
            if down_rate != 1:
                sample_data = down_sample(data[start: start+length], down_rate=down_rate)
            else:
                sample_data = data[start: start+length]
            wave = wavelet_transform(sample_data, level=level)
            samples.append(wave)
    elif channel_num == 2:
        for i in range(sample_num):
            for j in range(len(data)):
                start = random.randint(sample_start, sample_end)
                temp = []
                if down_rate != 1:
                    lead = down_sample(data[j][start: start+length], down_rate=down_rate)
                else:
                    lead = data[j][start: start+length]
                temp.append(lead)
            wave = wavelet_transform(temp, level=level)
            samples.append(wave)
    else:
        print("Input data shape is wrong!")
        raise TypeError
    return samples


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


def wavelet_transform(lead, level):
    """

    :param lead: list 心跳时间序列
    :param level: int 小波变换阶数
    :return: data_dict dict 小波变换后数据字典
    """
    data_dict = {}
    for i in range(level):
        if i == 0:
            data_dict['CA'+str(level)] = []
        data_dict['CD'+str(level-i)] = []
    if len(np.shape(lead)) == 2:
        for i in range(len(lead)):
            waves = pywt.wavedec(lead[i], wavelet='haar', level=level)
            for j in range(len(waves)):
                if j == 0:
                    data_dict['CA'+str(level-j)].append(waves[j].tolist())
                else:
                    data_dict['CD'+str(level-j+1)].append(waves[j].tolist())
        return data_dict
    elif len(np.shape(lead)) == 1:
        waves = pywt.wavedec(lead, wavelet='haar', level=level)
        for j in range(len(waves)):
            if j == 0:
                data_dict['CA' + str(level - j)] = waves[j].tolist()
            else:
                data_dict['CD' + str(level - j + 1)] = waves[j].tolist()
        return data_dict
    else:
        print("Input shape is Wrong!")
        raise TypeError


def write_files(data, write_path, tag, num):
    """
    写样本
    :param data: list 样本集
    :param write_path: string 写文件路径
    :param tag: string 标签
    :param num: int 文件名index
    :return: num int 文件名index
    """
    for value in data:
        file_name = str(num) + '_' + str(tag) + '.json'
        num += 1
        file_path = os.path.join(write_path, file_name)
        g = open(file_path, 'w')
        json.dump(value, g, indent=4)
        g.close()
    return num


if __name__ == "__main__":
    para_dict = {'sample_seconds': 8, 'train_seconds': 40, 'test_seconds': 40, 'train_samples': 100,
                 'test_samples': 100, 'val_samples': 50, 'attention_layer': 3}

    original_path = '/home/lab307/chenbin/data/original_data/identification/mit_original_data/'
    sample_path = '/home/lab307/chenbin/data/original_data/identification/train_model/'
    log_path = '/home/lab307/chenbin/data/original_data/identification/log/'
    write_path = '/home/lab307/chenbin/data/original_data/identification/result/'
    data_process(original_path, sample_path, para_dict)