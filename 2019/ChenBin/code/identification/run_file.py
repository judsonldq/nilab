# coding: utf-8

import sys
sys.path.append('./')
from data.process import data_process
from model.attention_model import model_flow
from model.attention_test_model import test_model_flow



if __name__ == "__main__":
    len_list = [6]
    # val_list = [40, 80]
    for i in range(len(len_list)):
        para_dict = {'sample_seconds': 2, 'train_seconds': 40, 'test_seconds': 40, 'train_samples': 300,
                     'test_samples': 300, 'val_samples': 150, 'attention_layer': 0}
        # para_dict['sample_seconds'] = len_list[i]
        # para_dict['val_samples'] = val_list[i]
        # para_dict['train_seconds'] = para_dict['test_seconds'] = int(45 * len_list[i] / 2)
        # original_path = './identification_data/'
        original_path = '../origin_data/' # 原始数据路径
        sample_path = './train_model/'
        log_path = './log/'
        write_path = './result/'
        data_process(original_path, sample_path, para_dict)
        model_flow(sample_path, log_path, write_path, para_dict)
        test_model_flow(sample_path, log_path, write_path, para_dict)
