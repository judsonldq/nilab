# -*- encoding: utf-8 -*-

import sys
sys.path.append("./")
from model.experiment_single_beat import experiment_find_offset
from model.experiment_subwaves import experiment_subwaves
# from model.experiment_tri_beat import experiment_subwaves
from model.experiment_hand_feature_tribeat_subwaves import experiment_hand_feature
from data_process.generate_samples import get_single_beat
from model.attention_model import triple
from model.attention_single import single
from model.attention_wave import wave

"""
if __name__ == "__main__":
    read_path = "/home/lab307/chenbin/data_process/original_data/MITDB/"
    write_path = "/home/lab307/chenbin/data_process/experiment/dynamic/"
    record_path = "/home/lab307/chenbin/paper/experiment/new_find_offset_last/"
    offset_list = ["0", "10", "20", "30",  "40", "50", "60", "70",  "80", "90",  "100"]
    # at last set offset to 0.35
    for offset in offset_list:
        samples(origin_path=read_path,
                write_path=write_path,
                rate=-int('100')*0.01)
        experiment_find_offset(offset=offset,
                               read_path=write_path,
                               write_path=record_path)
"""


if __name__ == "__main__":
    read_path = "/home/lab307/chenbin/data/original_data/MITDB/"
    write_path = "/home/lab307/chenbin/data/experiment/dynamic/"
    record_path = "/home/lab307/chenbin/paper/experiment/classification/"
    weight_path = '/home/lab307/chenbin/paper/experiment/weights/classification/'
    # at last set offset to 0.35
    offset = '35'
    # get_single_beat(origin_path=read_path,
    #                 write_path=write_path,
    #                 rate=int(offset)*0.01)
    # experiment_find_offset(offset=offset,
    #                        read_path=write_path,
    #                        write_path=record_path)
    # experiment_tri_beat(read_path=write_path,
    #                     write_path=record_path)
    # experiment_subwaves(read_path=write_path,
    #                     write_path=record_path)
    # experiment_hand_feature(read_path=write_path,
    #                         write_path=record_path)
    # experiment_hand_feature(write_path, record_path, weight_path)
    # triple()
    wave()
    single()
