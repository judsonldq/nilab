#-*- coding: UTF-8 -*-
import csv

f = open(u'real_sim.txt', "w",encoding= 'utf-8')                 #保存真实的相似专利

csv_reader = csv.reader(open('..\\相关数据\\输入集合.csv', 'r', encoding='gb18030'))

for line in csv_reader:
    lines=str(line[5]).split(" ")
    f.write(str(lines)+'\n')
f.close()