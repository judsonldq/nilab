# -*- encoding:utf-8 -*-

# coding: utf-8
import os
import shutil
import matplotlib.pyplot as plt

def generate_pictures(folder_path, save_path):
    file_list = os.listdir(folder_path)
    make_folder(save_path)
    for file in file_list:
        plot_picture(folder_path, save_path, file)


def plot_picture(folder_path, save_path, file):
    file_name = os.path.join(folder_path, file)
    data = read_file(file_name)
    fig = plt.figure(figsize=(25,25))
    ax_list = []
    for i in range(len(data)):
        ax_list.append('ax_'+str(i+1))
        ax_list[i] = plt.subplot(12,1,i+1)
        ax_list[i].plot(data[i], c='k')
        # ax_list[i].plot(data[i][5000:15000], c='k')
    name = file.split('.')[0] + '.png'
    save_file = os.path.join(save_path, name)
    # print(save_file)
    plt.savefig(save_file, bbox_inches = 'tight')
    plt.close()

def read_file(file_name):
    data = []
    f = open(file_name,'r')
    for line in f.readlines():
        temp = [round(float(i), 6) for i in line.strip().split(',')]
        data.append(temp)
    return data

def make_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def rename_file(read_path, write_path):
    """
    start file name form 0
    :param read_path: str read folder path
    :param write_path: str save folder path
    :return: None
    """
    make_folder(path=write_path)
    file_list = os.listdir(read_path)
    for file in file_list:
        num = int(file.split('.')[0]) - 1
        name = str(num) + '.' + file.split('.')[1]
        old_file_path = os.path.join(read_path, file)
        new_file_path = os.path.join(write_path, name)
        shutil.copyfile(old_file_path, new_file_path)


if __name__ == '__main__':
    # read_path = '/home/lab307/chenbin/data/original_data/CCDD/210/'
    read_path = '/home/lab307/sunhuan/ECG/identification/data/PTB/ptb_selected/'
    write_path = '/home/lab307/sunhuan/ECG/identification/data/PTB/ptb_plot/'
    # rename_file(read_path=read_path, write_path=write_path)
    folder_path = '/home/lab307/chenbin/data/original_data/CCDD/read_data_250/'
    save_path = '/home/lab307/chenbin/data/plot/210/'
    generate_pictures(read_path, write_path)