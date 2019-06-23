#-*- coding: UTF-8 -*-
from gensim.models import word2vec
from gensim import models
import math
from scipy import linalg,dot
import heapq
import csv
import re
import jieba
import jieba.posseg as pseg
import numpy as np
from scipy.optimize import linear_sum_assignment
from pulp import *
import time

jieba.load_userdict("..\\相关数据\\发现新词去重后-优化.txt")
stopwords = {}.fromkeys([line.rstrip() for line in open("..\\相关数据\\停用词表.txt",'r', encoding='UTF-8')])     #停用词路径

BD2 = "[(|<]{1}[0-9]*[a-zA-Z]*[)|>]{1}"
BD1 = "[\u0060|\u0021-\u002c|\u002e-\u002f|\u003a-\u003f|\u2200-\u22ff|\uFB00-\uFFFD|\u2E80-\u33FF|<sub>|</sub>|'|“|”]"

# 全集数据：输入+输出
# csv_reader = csv.reader(open('F:\\基于异质信息网络的专利分析 - new\\相似度计算\\新的专利数据\\专利数据_test.csv','r',encoding='gb18030'))      ##原始数据
csv_reader1 = csv.reader(open('..\\相关数据\\输入集合.csv','r',encoding='gb18030', errors='ignore'))      ##原始数据
csv_reader2 = csv.reader(open('..\\相关数据\\输出集合.csv','r',encoding='gb18030', errors='ignore'))      ##原始数据

#csv_reader1 = csv.reader(open('/Users/chenzelong/Desktop/做图/输入集合.csv','r',encoding='utf8', errors='ignore'))      ##原始数据
#csv_reader2 = csv.reader(open('/Users/chenzelong/Desktop/做图/输出集合.csv','r',encoding='utf8', errors='ignore'))      ##原始数据


model = models.Word2Vec.load('..\\词向量训练\\word2vec_all.model')

def split_word(string):
    final = ''
    seg_list = re.sub(BD2, '',string)
#    seg_list = re.sub(BD1, '',seg_list)                                  ############去标点符号

    seg_list = jieba.posseg.cut(seg_list)
    for seg in seg_list:
        final = final + " " +seg.word
    seg_list = final
    final = ''

    seg_list =seg_list.split()                                        ###############去空格
    n = len(seg_list)
    for i in range(n):
        # 删除那些仅由英语数字标点组成的字符串
#        pattern = re.compile(r'[0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*[0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*$')
        pattern = re.compile(r'[A-Za-z0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*[A-Za-z0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*$')

        match = pattern.match(seg_list[i])
        if match:
            flag = False
        else:
            flag = True        
        if seg_list[i] not in stopwords and flag:                               ##############去停用词
            seg_list[i] = re.sub(BD1, '',seg_list[i])
            if ("nm" in seg_list[i] and seg_list[i].index("nm") == 0) or ("ml" in seg_list[i] and seg_list[i].index("ml") == 0):
                seg_list[i] = seg_list[i][2:]
            if seg_list[i].isnumeric()  or len(seg_list[i]) == 1:
                seg_list[i] = ""
            final += seg_list[i]
            final += " "
    final = final.split()
    return final





def creat_patent_dic(csv_reader):
    patents = {}
    PatToSimPats = {}
    PatentClass = {}
    for line in csv_reader:
#        print(line)
        apatent = []
        sim_patent = []
        patent_name = line[0]
        patent_name_list = split_word(patent_name)
        
        principal_claim = line[1]
        principal_claim_list = re.split(r'[,|.|;|，|。|；]',principal_claim)
        i = 0
        while i < len(principal_claim_list):
            principal_claim_list[i] = split_word(principal_claim_list[i])
            if len(principal_claim_list[i]) == 0:
                principal_claim_list.remove(principal_claim_list[i])
            else:
                i = i+1
        apatent.extend([patent_name_list])
        apatent.extend(principal_claim_list)
#        apatent.extend(function_list)
        MainClass = line[3]
        sim_patent = line[5]
        sim_patent = sim_patent.split()
        if patent_name not in patents:
            patents[patent_name] = apatent
            PatToSimPats[patent_name] = sim_patent
            PatentClass[patent_name] = MainClass
        elif patent_name + " " not in patents:
            patents[patent_name+" "] = apatent
            PatToSimPats[patent_name+" "] = sim_patent
            PatentClass[patent_name+" "] = MainClass
        elif patent_name + "  " not in patents:
            patents[patent_name+"  "] = apatent
            PatToSimPats[patent_name+"  "] = sim_patent
            PatentClass[patent_name+"  "] = MainClass
        elif patent_name + "   " not in patents:
            patents[patent_name+"   "] = apatent
            PatToSimPats[patent_name+"   "] = sim_patent
            PatentClass[patent_name+"   "] = MainClass
        elif patent_name + "    " not in patents:
            patents[patent_name+"    "] = apatent
            PatToSimPats[patent_name+"    "] = sim_patent
            PatentClass[patent_name+"    "] = MainClass
        elif patent_name + "     " not in patents:
            patents[patent_name+"     "] = apatent
            PatToSimPats[patent_name+"     "] = sim_patent
            PatentClass[patent_name+"     "] = MainClass
    return patents,PatToSimPats,PatentClass



def DTW(grid):
    if len(grid) and len(grid[0]):
        k = 1
        m = len(grid[0])
        n = len(grid)
        min__ = min(n,m)
        max__ = max(n,m)
        dp = [[0 for _ in range(m)] for _ in range(n)]
        summ = 0
        sumn = 0
        for i in range(n):
            dp[i][0] = summ + grid[i][0]
            summ = dp[i][0]
        for j in range(m):
            dp[0][j] = sumn + grid[0][j]
            sumn = dp[0][j]
        for line in range(1, n):
            for row in range(1, m):
                dp[line][row] = min(dp[line - 1][row], dp[line][row - 1],dp[line - 1][row - 1])+grid[line][row]
                k = k+1
        return (1/k)*dp[n - 1][m - 1]
    else:
        return 999999999


def position(k,l):
    if l%2 == 0:
        if k <= l//2:
            val = (1+l/2)*(l/2)
            p = k/val
        if k > l//2:
            k = l-k
            val = (1+l/2)*(l/2)
            p = k/val
    if l%2 != 0:
        if k <= (l//2+1):
            val = (1+l//2)*(l//2) + (l//2+1)
            p = k/val
        else:
            k = l-k +1
            val = (1+l//2)*(l//2) + (l//2+1)
            p = k/val  
    return p 

def delta1(i,j,l1,l2):
    position1 = position(i,l1)
    position2 = position(j,l2)
    idx = min(1/l1,1/l2)
#    idx = [(1/l1)+(1/l2)]/2
    mean = abs(position1-position2)
    if mean == 0:
        mean = idx
    return mean


def sentence_similarity(patentA_sent,patentB_sent): # 衡量句子和句子之间的相似度，越小越好
    cost_ab = []
    for i in range(len(patentA_sent)):
        dis = []
        v1 = model.wv[patentA_sent[i]]
        for j in range(len(patentB_sent)):
            v2 = model.wv[patentB_sent[j]]
            delta_ij = delta1(i,j,len(patentA_sent),len(patentB_sent))
            d_ij = np.linalg.norm(v1 - v2) # 欧氏距离
            dis.append(delta_ij*d_ij)
        cost_ab.append(dis)

    cost1 = []
    cost2 = []
    n1 = len(cost_ab)
    n2 = len(cost_ab[0])
    # aList.index( 'xyz' )
    for i in range(n1):
        temp = []
        alist = sorted(cost_ab[i])
        #    print(cost[i])
        for j in range(n2):
            #        print(cost[i][j])
            idx = alist.index(cost_ab[i][j]) + 1
            #        print(idx)
            temp.append(idx)
        cost1.append(temp)
#    print(cost1)
    cost_ab = np.array(cost_ab)
    cost_ba = cost_ab.T
    # print(cost_ba)
    cost_ba = cost_ba.tolist()
    # print(cost_ba)
    n1 = len(cost_ba)
    n2 = len(cost_ba[0])
    # aList.index( 'xyz' )
    for i in range(n1):
        temp = []
        alist = sorted(cost_ba[i])
        #    print(cost[i])
        for j in range(n2):
            #        print(cost[i][j])
            idx = alist.index(cost_ba[i][j]) + 1
            #        print(idx)
            temp.append(idx)
        cost2.append(temp)
#    print(cost2)

    cost = []
    for i in range(len(cost1)):
        cost_ = []
        for j in range(len(cost2)):
            #        print(cost1[i][j],cost2[j][i])
            val1 = cost1[i][j]
            val2 = cost2[j][i]
            val = (val1 ** 2 + val2 ** 2) / (val1 + val2)
            cost_.append(val)
        cost.append(cost_)
#    print(cost)
    cost = np.array(cost)
    r, c = linear_sum_assignment(cost)  # 得到最佳分配下的行列索引值
#    print(r, c)
    r = r.tolist()
    c = c.tolist()
    res = 0
    for i in range(len(r)):
        res = res + cost[r[i]][c[i]]*cost_ab[r[i]][c[i]]
#        print(cost[r[i]][c[i]])


    val = max(len(patentA_sent),len(patentB_sent))/min(len(patentA_sent),len(patentB_sent))
    return val*res



# PatentClass

# patents


def Patent_similarity(Apatents, Bpatents):  # 衡量专利之间的相似度，越小越好
    sim_dic = {}
    for patentA in Apatents:
        for patentB in Bpatents:
            #            if APatentClass[patentA] != BPatentClass[patentB]:
            #                continue
            if patentA != patentB:
                patentA_list = Apatents[patentA]
                patentB_list = Bpatents[patentB]
                #                print(patentA,Apatents[patentA])
                #                print(patentB,Bpatents[patentB])
                if len(patentA_list[0]) == 0 or len(patentB_list[0]) == 0:
                    #                    sim_title = 0
                    sim_title = "a"
                else:
                    sim_title = sentence_similarity(patentA_list[0],
                                                        patentB_list[0])  # 计算标题的相似度，patentA_list[0]是标题分词后存下来的列表
                patentA_Sentence_weight = []
                patentB_Sentence_weight = []
                patent_distance = []
                #                for k in range(1,len(patentB_list)):
                #                    patentB_Sentence_weight.append(len(patentB_list[k]))
                if len(patentA_list) > 1 and len(patentB_list) > 1:
                    for i in range(1, len(patentA_list)):
                        patentA_Sentence_weight.append(len(patentA_list[i]))
                        sent_distance = []
                        for j in range(1, len(patentB_list)):
                            if i == 1:
                                patentB_Sentence_weight.append(len(patentB_list[j]))
                            sim_sent = sentence_similarity(patentA_list[i], patentB_list[j])
                            sent_distance.append(sim_sent)
                        patent_distance.append(sent_distance)
                patent_distance = np.array(patent_distance)
                sim_art = DTW(patent_distance)
                if sim_title == "a":
                    sim_title = sim_art
                if patentA not in sim_dic:
                    dic = {}
                    dic[patentB] = [sim_title, sim_art]
                    sim_dic[patentA] = dic
                else:
                    dic = sim_dic[patentA]
                    dic[patentB] = [sim_title, sim_art]
                    sim_dic[patentA] = dic
    return sim_dic



def TopKPatents(PatentToPatent_all,k):
    alist = []
    count = 0
    for key,val in PatentToPatent_all:
        if key.strip() not in alist:
            count += 1
            alist.append(key.strip())
        if count>=k:
            break
    return alist




def  Recommend_Effect(alist,simPat):
    FP, FN, TP = 0, 0, 0
    for p1 in alist:
        p1 = p1.strip()
        if p1 in simPat:
            TP = TP + simPat.count(p1)
        elif p1 not in simPat:
            FP = FP + 1
    for p2 in simPat:
        if p2 not in alist:
            FN = FN + 1
    A = TP/(TP+FP)
    B = TP/(TP+FN)
    precision = max(A, B)
    return precision

start = time.clock()  # 程序开始时间

Apatents,APatToSimPats,APatentClass = creat_patent_dic(csv_reader1)
Bpatents,BPatToSimPats,BPatentClass = creat_patent_dic(csv_reader2)


sim_dic = Patent_similarity(Apatents,Bpatents)


def func(sim_dic):
    k = 0
    precision_sum = 0
    for patent in sim_dic:
        PatentToPatent_all = {}
        PatentToPatent = sim_dic[patent]
        for ToPatent in PatentToPatent:
            PatentToPatent_all[ToPatent] = PatentToPatent[ToPatent]
        PatentToPatent_all_05 = sorted(PatentToPatent_all.items(),key = lambda PatentToPatent_all:0.5*PatentToPatent_all[1][0]+0.5*PatentToPatent_all[1][1],reverse=False)
        alist = TopKPatents(PatentToPatent_all_05,5)    # 推荐出来的相似专利的结果
        simPat = APatToSimPats[patent]            # 真实结果
#        print(patent,alist)  # 推荐结果
#        print(patent,simPat) # 真实结果
        precision = Recommend_Effect(alist, simPat)

        k = k + 1
        precision_sum = precision_sum + precision

    #print(precision_sum)
    #print(k)
    print(precision_sum/k)

func(sim_dic)
elapsed = (time.clock() - start)
print("Time used:",elapsed)












