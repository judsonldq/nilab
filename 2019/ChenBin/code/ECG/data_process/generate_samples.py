# coding: utf-8
import os
import sys
import time
import random

sys.path.append("./")
from tool.helper import make_folder, cut_minutes, label_transform, add_zero, math_feature, down_sample
from conf.config import LENGTH, VAL_NUMBER, KEY_LIST


def get_single_beat(origin_path, write_path, rate=0.35):
    """
    生成训练样本
    :param origin_path: str 源数据地址
    :param write_path: str 训练数据地址
    :param rate: float 偏移百分比
    :return: None
    """
    index_train = 245
    index_test = 0
    # 生成对应文件夹路径
    origin_data_path = origin_path + 'read_data/'
    origin_atr_path = origin_path + 'read_atr/'
    write_training_path = write_path + 'train/'
    write_test_path = write_path + 'test/'
    write_val_path = write_path + 'val/'
    make_folder(write_training_path)
    make_folder(write_test_path)
    make_folder(write_val_path)
    # 分两批处理数据
    # first_block = [115]
    # second_block = []
    first_block = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 121, 122, 123, 124]
    second_block = [200, 201, 202, 203, 205, 207, 208, 209, 210, 212,
                    213, 214, 215, 219, 220, 221, 222, 223, 228, 230,
                    231, 232, 233, 234]
    # 处理第一部分数据
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "开始切分第一部分数据...")
    for patient in first_block:
        # print(str(patient)+' is start...')
        data_name = str(patient) + 'data.csv'
        atr_name = str(patient) + 'label.csv'
        patient_data_path = origin_data_path + data_name
        patient_atr_path = origin_atr_path + atr_name
        data, data_time = load_data(file_path=patient_data_path)
        label, label_time = load_data(file_path=patient_atr_path)
        R_time = find_Rtime(label_time=label_time, data_time=data_time)
        chip_beat = chip_samples(data_list=data, Rtime=R_time, label=label, max_length=LENGTH, rate=rate)
        index_test = generate_test_dataset(data=chip_beat,
                                           test_index=index_test, test_path=write_test_path)
        # print(str(patient) + ' is end.')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "第一部分数据切分完成！")
    # 从第一部分数据中抽样245个数据
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "开始抽样245个训练数据样本...")
    generate_245_global_samples(read_path=write_test_path, write_path=write_training_path)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "245个训练样本抽样完成！")

    # 处理第二部分数据
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "开始切分第二部分数据...")
    for patient in second_block:
        # print(str(patient) + ' is start...')
        data_name = str(patient) + 'data.csv'
        atr_name = str(patient) + 'label.csv'
        patient_data_path = origin_data_path + data_name
        patient_atr_path = origin_atr_path + atr_name
        data, data_time = load_data(file_path=patient_data_path)
        label, label_time = load_data(file_path=patient_atr_path)
        R_time = find_Rtime(label_time=label_time, data_time=data_time)
        time_index = cut_minutes(label_time=label_time, n=5)
        chip_beat = chip_samples(data_list=data, Rtime=R_time, label=label, max_length=LENGTH, rate=rate)
        # 生成训练集， time_index - 1是因为chip_samples()里 Rtime 从第1个开始
        index_train = generate_training_dataset(data=chip_beat[:time_index-1],
                                                train_index=index_train, training_path=write_training_path)
        index_test = generate_test_dataset(data=chip_beat[time_index-1:],
                                           test_index=index_test, test_path=write_test_path)
        # print(str(patient) + ' is end.')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "第二部分数据切分完成！")
    # 生成验证集
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "生成验证数据...")
    get_val_dataset(source_path=write_test_path, val_path=write_val_path, num=VAL_NUMBER)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "验证数据生成完成！")


def generate_245_global_samples(read_path, write_path):
    """
    随机抽样245个心拍
    :param read_path: str 读取样本路径
    :param write_path: str 写样本路径
    :return: None
    """
    sample_index = 0
    sample_dict = {'0': 75, '1': 75, '2': 75, '3': 13, '4': 7}
    file_class_dict = {'0': [], '1': [], '2': [], '3': [], '4': []}
    file_list = os.listdir(read_path)
    for file in file_list:
        tag = file.split('_')[1].split('.')[0]
        file_class_dict[tag].append(file)
    for key in file_class_dict.keys():
        sample_file_list = random.sample(file_class_dict[key], sample_dict[key])
        for file in sample_file_list:
            file_read_path = os.path.join(read_path, file)
            sample_name = str(sample_index) + '_' + file.split('_')[1]
            file_write_path = os.path.join(write_path, sample_name)
            sample_index += 1
            f = open(file_read_path, 'r')
            text = f.read()
            f.close()
            with open(file_write_path, 'w') as g:
                g.write(text)


def generate_training_dataset(data, train_index, training_path):
    """
    生成训练集
    :param data: list 训练集数据
    :param train_index: int 样本编号
    :param training_path: str 训练集文件路径
    :return: index int 样本编号
    """
    train_index = write_files(index=train_index, all_beat_list=data, write_path=training_path)
    return train_index


def generate_test_dataset(data, test_index, test_path):

    test_index = write_files(index=test_index, all_beat_list=data, write_path=test_path)
    return test_index


def get_val_dataset(source_path, val_path, num):
    """
    生成验证数据集
    :param source_path: string 验证数据集源的文件路径
    :param val_path: string 验证数据集保存路径
    :param num: int 验证数据集的数目
    :return: None
    """
    file_list = os.listdir(source_path)
    val_list = random.sample(file_list, num)
    for files in val_list:
        old_path = os.path.join(source_path, files)
        new_path = os.path.join(val_path, files)
        os.rename(old_path, new_path)


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


def chip_samples(data_list, Rtime, label, max_length, rate=0.35, down_rate=3, p_para=0.25, qrs_para=0.25, overlap_para=0.03):
    """
    切分心跳，获取特征
    :param data_list: list 原始数据
    :param Rtime: list 标签下标
    :param label: list 标签
    :param max_length: int 最大长度设定
    :param rate: float 偏移量
    :param p_para: float p波占比
    :param qrs_para: float qrs波占比
    :param overlap_para: float 重叠比例
    :return: all_beat_list: 元素是单个心跳的所有输入
    """
    all_beat_list = []
    length = len(data_list)
    down_max = max_length // down_rate
    for i in range(1, len(Rtime) - 1):
        beat = []
        single_length = Rtime[i + 1] - Rtime[i]
        if single_length > max_length:
            single_length = max_length
        down_single = single_length // down_rate
        # 设置偏移量
        shift = int(single_length * rate)
        # 对Rtime标志位判断
        # 判断结尾处是不是比最后一个心跳短
        if length < Rtime[i] + single_length - shift:
            single_beat = down_sample(data_list[-single_length - 1:-1], down_rate)
            print('the loop ending is less than %s' % single_length)
        # 判断开头处是不是比第一个心跳短
        elif Rtime[i] < shift:
            single_beat = down_sample(data_list[:single_length], down_rate)
            print('the beginning is less than %s' % shift)
        # 正常心跳切分
        else:
            single_beat = down_sample(data_list[Rtime[i]-shift: Rtime[i]+single_length-shift], down_rate)
        single_feature = math_feature(single_beat)
        # 获取三倍拍长
        if length < Rtime[i] + 2*single_length - shift:
            tri_beat = down_sample(data_list[-3*single_length - 1:-1], down_rate)
        elif Rtime[i] < shift + single_length:
            tri_beat = down_sample(data_list[:3*single_length], down_rate)
        else:
            tri_beat = down_sample(data_list[Rtime[i]-single_length-shift: Rtime[i]+2*single_length-shift], down_rate)
        # 获取子拍
        one = int(p_para * down_single)
        two = int(qrs_para * down_single)
        three = len(single_beat) - one - two
        t_para = 1 - p_para - qrs_para
        overlap = int(overlap_para * down_single)
        p_wave = single_beat[:one + overlap]
        p_feature = math_feature(p_wave)
        p_wave = add_zero(p_wave, (p_para + overlap_para) * down_max)
        if len(p_wave) != 28:
            print("p wave:", len(p_wave))
        qrs_wave = single_beat[one - overlap: one + two + overlap]
        qrs_feature = math_feature(qrs_wave)
        qrs_rate = round((rate-p_para+overlap_para)/(qrs_para+2*overlap_para), 3)
        qrs_wave = add_zero(qrs_wave, (qrs_para + 2 * overlap_para) * down_max, mode='center', rate=qrs_rate)
        # qrs_wave = add_zero(qrs_wave, (qrs_para + 2 * overlap_para) * down_max)
        if len(qrs_wave) != 31:
            print("qrs wave:", len(qrs_wave))
        t_wave = single_beat[three - overlap:]
        t_feature = math_feature(t_wave)
        t_wave = add_zero(t_wave, (t_para + overlap_para) * down_max)
        if len(t_wave) != 53:
            print("t wave:", len(t_wave))
        # 补零
        single_beat = add_zero(single_beat, down_max, mode='center', rate=rate)
        tri_rate = round((1+rate)/3, 3)
        tri_beat = add_zero(tri_beat, max_length, mode='center', rate=tri_rate)
        # tri_beat = add_zero(tri_beat, max_length)
        beat.append(single_beat)
        beat.append(tri_beat)
        beat.append(p_wave)
        beat.append(qrs_wave)
        beat.append(t_wave)

        # 补充其他特征
        # max_data = max(single_beat)
        # min_data = min(single_beat)
        # duration = max_data - min_data
        # abs_max = max(abs(max_data), abs(min_data))
        # 心跳长度特征
        last_RR = Rtime[i] - Rtime[i - 1]
        now_RR = Rtime[i+1] - Rtime[i]
        try:
            next_RR = Rtime[i + 2] - Rtime[i + 1]
        except IndexError:
            next_RR = now_RR
        last_diff_ratio = round(last_RR / now_RR - 1, 6)
        next_diff_ratio = round(next_RR / now_RR - 1, 6)
        compare_ratio = round(last_RR / next_RR - 1, 6)
        sum_ratio = round(abs(last_RR - next_RR) / (1.0 * (last_RR + next_RR)), 6)
        first_ratio = round(now_RR / (1.0 * (last_RR + next_RR)), 6)
        length_feature = [last_diff_ratio, next_diff_ratio, compare_ratio, sum_ratio, first_ratio, now_RR]
        # last_diff_ratio = round(last_RR / now_RR, 6)
        # next_diff_ratio = round(next_RR / now_RR, 6)
        # length_feature = [last_diff_ratio, next_diff_ratio]
        feature = []
        feature.extend(single_feature)
        feature.extend(p_feature)
        feature.extend(qrs_feature)
        feature.extend(t_feature)
        feature.extend(length_feature)
        beat.append(feature)
        # 添加label特征
        beat.append(label[i])
        # single_beat: list 第一个元素：单个心跳数组
        #                   第二个元素：三连心跳数组
        #                   第三、四、五个元素：p， qrs， t 波数组
        #                   第五个元素：手动特征数组
        #                   第六个元素：标签
        all_beat_list.append(beat)
    return all_beat_list


def write_files(index, all_beat_list, write_path, classes=5, key_list=KEY_LIST):
    for beat in all_beat_list:
        tag = beat.pop()
        tag = label_transform(tag, classes)
        if tag is None:
            continue
        file_name = str(index) + '_' + str(tag) + '.csv'
        index += 1
        file_path = os.path.join(write_path, file_name)
        string = []
        for thing in beat:
            thing = [str(element) for element in thing]
            string.append(','.join(thing))
        with open(file_path, 'w') as f:
            for k in range(len(string)):
                f.write(key_list[k] + ':' + string[k] + '\n')
    return index


if __name__ == "__main__":
    org_path = '/home/lab307/chenbin/data/original_data/MITDB/'
    wri_path = '/home/lab307/chenbin/data/experiment/dynamic/'
    get_single_beat(origin_path=org_path,
                    write_path=wri_path)

