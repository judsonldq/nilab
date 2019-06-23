#-*- coding: UTF-8 -*-
import csv
import string
import jieba
import jieba.posseg as pseg
import re


alist = ["n","vn","v","j","nz","l","s","ng","f","nr","m","vg","nz","ng","x","q","d","eng"]

BD2 = "[(|<]{1}[0-9]*[a-zA-Z]*[)|>]{1}"
BD1 = "[\u0060|\u0021-\u002c|\u002e-\u002f|\u003a-\u003f|\u2200-\u22ff|\uFB00-\uFFFD|\u2E80-\u33FF]"

jieba.load_userdict("..\\相关数据\\发现新词去重后-优化.txt")

csv_reader = csv.reader(open('..\\相关数据\\输入集合.csv','r',encoding='gb18030', errors='ignore'))      ##原始数据
#csv_reader = csv.reader(open('C:\\Users\\czl\\Desktop\\基于异质信息网络的专利分析 - new\\构造测试集\\输出集合.csv','r',encoding='gb18030', errors='ignore')) 
#csv_reader = csv.reader(open('/Users/luw/Desktop/毕设/新数据集/输出集合_全.csv','r',encoding='gb18030'))      ##原始数据
#csv_reader = csv.reader(open('/Users/luw/Desktop/毕设/输入集合.csv','r'))      ##原始数据

stopwords = {}.fromkeys([line.rstrip() for line in open("..\\相关数据\\停用词表.txt",'r', encoding='UTF-8')])     #停用词路径
#stopwords = {}.fromkeys([line.rstrip() for line in open("/Users/luw/Desktop/毕设/停用词表.txt")])     #停用词路径

f = open(u'output1.txt', "w+",encoding= 'utf-8')                 #保存去停用词，标点符号，空格后的数据
#f = open(u'/Users/luw/Desktop/毕设/output1.txt', "w+")                 #保存去停用词，标点符号，空格后的数据

for line in csv_reader:                                                  ########## 数据从CSV文档开始读 要注意读文档的哪一列 成果是line[2] 需求是line[1]
    final = ''
    lines = str(line[1]).replace('\n','')
#    seg_list = re.sub(BD2, '',lines)
#    seg_list = re.sub(BD1, '',seg_list)                                  ############去标点符号

    seg_list = jieba.posseg.cut(lines)
    for seg in seg_list:
        if seg.flag in alist:
#        print(seg.word,seg.flag)
            final = final + " " +seg.word
    seg_list = final

    seg_list = ' '.join(re.findall(u'[\u4e00-\u9fff]+', seg_list))     ############去标点符号
    seg_list = seg_list.split()

    final = ''
#    seg_list = ' '.join(re.findall(u'[\u4e00-\u9fff]+', seg_list))     ############去标点符号

    n = len(seg_list)
    for i in range(n):
        if seg_list[i] not in stopwords:                               ##############去停用词
            final += seg_list[i]
            final += " "
    f.write(final + '\n')

f.close()

