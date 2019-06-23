#-*- coding: UTF-8 -*-
import csv

f = open(u'num_name.txt', "w+",encoding= 'utf-8')                 #保存标号和专利名称

csv_reader = csv.reader(open('..\\相关数据\\输出集合.csv','r',encoding='gb18030'))
#n=0
for line in csv_reader:
#    n=n+1
    lines=str(line[0])
#    f.write(str(n))
    f.write(lines+'\n')

f.close()
