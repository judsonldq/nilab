# -*- encoding:utf-8 -*-
import os
import shutil

def make_folder(path):
    """
    重新生成某个文件夹
    :param path: str 文件夹路径
    :return: None
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


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
                return 2 ** (i+bias+2)
            else:
                pass
    else:
        for i in range(8, length):
            if length < 2 ** i:
                return 2 **(i+bias)
            else:
                pass
