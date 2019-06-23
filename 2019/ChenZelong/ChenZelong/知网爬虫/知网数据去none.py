import csv
# 基于异质信息网络的专利分析\知网数据\知网爬虫最最近
csv_reader = csv.reader(open('F:\\基于异质信息网络的专利分析\\知网数据\\知网爬虫最最近\\电子.csv','r',encoding='utf8'))      ##原始数据 gb18030
#打开文件，追加a
out = open('半导体去none后.csv','a',encoding='gb18030', newline='')
#设定写入模式
csv_write = csv.writer(out,dialect='excel')


for line in csv_reader: 
    alist = []
    lines1 = str(line[1]).replace('\n',' ') 
    lines2 = str(line[2]).replace('\n',' ') 
    lines3 = str(line[3]).replace('\n',' ')
    lines4 = str(line[4]).replace('\n',' ')
    lines5 = str(line[5]).replace('\n',' ')
    lines6 = str(line[6]).replace('\n',' ')
    lines7 = str(line[7]).replace('\n',' ')
    lines8 = str(line[8]).replace('\n',' ')
    lines9 = str(line[9]).replace('\n',' ')
    alist.append(lines1)
    alist.append(lines2)
    alist.append(lines3)
    alist.append(lines4)
    alist.append(lines5)
    alist.append(lines6)
    alist.append(lines7)
    alist.append(lines8)
    alist.append(lines9)
    if lines1 != "none" and lines2 != "none"and lines3 != "none"and lines4 != "none"and lines5 != "none"and lines6 != "none"and lines7 != "none"and lines8 != "none"and lines9 != "none":
        csv_write.writerow(alist)

#csv_reader.close()
#out.close()
