#-*- coding: UTF-8 -*-
import csv
import linecache

f = open(u'result4.txt', "w+",encoding= 'utf-8')

for line in open('real_sim.txt', 'r', encoding='utf-8'):
    line=str(line).split(",")
#    print(len(line))
    result3 = []
    for j in range(len(line)):
        for i in range(7389):
            get_count = linecache.getline('num_name.txt', i+1).replace("\n", "")
            if (line[j]).strip().replace("'","").replace("[","").replace("]","") == get_count:
                result3.append((i+1))
    #print(result3)
    f.write(str(result3))
    f.write("\n")
                #f.write(str(i+1)+" ")
    #f.write("\n")
f.close()




