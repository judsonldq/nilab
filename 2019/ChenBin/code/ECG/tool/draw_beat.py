# -*- encoding:utf-8 -*-
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# -*- encoding:utf-8 -*-
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    读取心跳数据和标签数据
    :param file_path: string 待读取文件的路径
    :return: data_process: list 读取的数据
             time: list 读取该数据的时间
    """
    fr = open(file_path, 'r')
    time = []
    data = []
    for lines in fr.readlines():
        temp = lines.strip().split(',')
        if len(temp) != 2:
            print("Wrong!")
        temp[0] = float(temp[0])
        temp[1] = float(temp[1])
        time.append(temp[0])
        data.append(temp[1])
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
    for i in range(len(data_time) - 1):
        if j < len(label_time):
            if data_time[i] <= label_time[j] < data_time[i + 1]:
                Rtime.append(i)
                j += 1
        else:
            break
    return Rtime


def get_fixed_length(data_list, Rtime, label, max_length, rate):
    all_beat_list = []
    length = len(data_list)
    for i in range(1, len(Rtime) - 1):
        single_length = Rtime[i + 1] - Rtime[i]
        if single_length > max_length:
            single_length = max_length
        # 设置偏移量
        shift = int(single_length * rate)
        # 切500的长度
        # 开头
        if Rtime[i] < 200:
            beat = data_list[:Rtime[i] + 300]
            beat.append(0)
            end = single_length
            beat.append(end)

        elif length - Rtime[i] < 300:
            beat = data_list[Rtime[i] - 200:]
            start = 200 - shift
            beat.append(start)
            beat.append(length)
        else:
            beat = data_list[Rtime[i] - 200:Rtime[i] + 300]
            start = 200 - shift
            end = start + single_length
            beat.append(start)
            beat.append(end)
        beat.append(label[i])
        all_beat_list.append(beat)
    return all_beat_list


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


def write_files(all_beat_list, write_path, classes=5):
    index = 0
    for beat in all_beat_list:
        tag = beat.pop()
        tag = label_transform(tag, classes)
        if tag is None:
            continue
        file_name = str(index) + '_' + str(tag) + '.png'
        index += 1
        file_path = os.path.join(write_path, file_name)
        end = beat.pop()
        start = beat.pop()
        length = end - start

        fig = plt.figure(figsize=(8, 8))

        first = np.arange(start)
        second = np.arange(start, end)
        third = np.arange(end, len(beat))
        before = beat[:start]
        now = beat[start:end]
        after = beat[end:]

        ax1 = fig.add_subplot(211)
        ax1.grid()
        ax1.plot(first,before,c='k',linewidth=1)
        ax1.plot(second,now,c='k',linewidth=3)
        ax1.plot(third,after,c='k',linewidth=1)
        ax1.axvline(start, color='r', linestyle='--')
        ax1.axvline(end, color='r', linestyle='--')

        samples = beat[start:end]
        one = int(0.25*length)
        two = int(0.25*length)
        overlap = int(0.06*length)
        p_wave = samples[:one+overlap]
        qrs_wave = samples[one-overlap//2:one+two+overlap//2]
        t_wave = samples[one+two-overlap:]
        x1 = np.arange(len(p_wave))
        x2 = np.arange(len(p_wave)+20, len(p_wave)+len(qrs_wave)+20)
        x3 = np.arange(len(p_wave)+len(qrs_wave)+40,len(p_wave)+len(qrs_wave)+len(t_wave)+40)

        ax2 = fig.add_subplot(212)
        ax2.grid()
        ax2.plot(x1, p_wave, c='k', linewidth=3)
        ax2.plot(x2, qrs_wave, c='k', linewidth=3)
        ax2.plot(x3, t_wave, c='k', linewidth=3)
        ax2.set_xlim([-20, len(p_wave)+len(qrs_wave)+len(t_wave)+60])

        plt.savefig(file_path)
        plt.close()


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


def draw_picture(origin_path, write_path, rate=0.45):
    origin_data_path = origin_path + 'read_data/'
    origin_atr_path = origin_path + 'read_atr/'
    file_list = os.listdir(origin_data_path)
    MAX_LENGTH = 300
    for file in file_list:
        num = file[:3]
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), num+"开始切分数据...")
        atr_name = num + 'label.csv'
        atr_path = os.path.join(origin_atr_path, atr_name)
        data_path = os.path.join(origin_data_path, file)
        picture_path = os.path.join(write_path, num)
        if os.path.exists(picture_path):
            shutil.rmtree(picture_path)
        os.mkdir(picture_path)
        labels, label_time = load_data(atr_path)
        data, data_time = load_data(data_path)
        Rtime = find_Rtime(label_time, data_time)
        flag = cut_minutes(label_time, 5)
        all_beat = get_fixed_length(data, Rtime[:flag],
                                    labels, MAX_LENGTH, rate)
        write_files(all_beat, picture_path)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), num + "完成切分数据！")


if __name__ == "__main__":
    read_path = "/home/lab307/chenbin/data_process/original_data/MITDB/"
    write_path = "/home/lab307/chenbin/data_process/experiment/split_sub_wave/0.25_0.25_0.5_0.06/"
    draw_picture(read_path, write_path, rate=0.35)