import csv
import re
import string
import jieba
import jieba.posseg as pseg
import re
from gensim import corpora, models, similarities
from numpy import *
import numpy as np
from nltk.cluster.util import cosine_distance

jieba.load_userdict('..\\相关数据\\发现新词去重后-优化.txt')  # 加载自定义词典
csv_reader1 = csv.reader(open('..\\相关数据\\输入集合.csv', 'r', encoding='gb18030'))  ##原始数据
csv_reader2 = csv.reader(open('..\\相关数据\\输出集合.csv', 'r', encoding='gb18030'))
stopwords = {}.fromkeys([line.rstrip() for line in open("..\\相关数据\\停用词表.txt", encoding='utf8')])  # 停用词路径

def tokenization(filename):
    result = []
    for line in open(filename, 'r',encoding='utf-8_sig'):
        line = line.split()
        result.append(line)
    return result
test_corpus = tokenization('output.txt')
dictionary = corpora.Dictionary(test_corpus)   # 建立词袋模型
vectors = [dictionary.doc2bow(text) for text in test_corpus]  #######统计词频 并用词频表示产品向量 [(0, 1), (1, 1), (2, 1), (3, 1),.....]
#print(vectors)
model = models.TfidfModel(vectors)  # 计算所有词的tfidf


def split_word(lines):
    alist = ["n", "vn", "v", "j", "nz", "l", "s", "ng", "f", "nr", "m", "vg", "nz", "ng", "x", "q", "d", "eng"]
    BD2 = "[(|<]{1}[0-9]*[a-zA-Z]*[)|>]{1}"
    BD1 = "[\u0060|\u0021-\u002c|\u002e-\u002f|\u003a-\u003f|\u2200-\u22ff|\uFB00-\uFFFD|\u2E80-\u33FF]"
    #for f in lines:  ########## 数据从CSV文档开始读 要注意读文档的哪一列 成果是line[2] 需求是line[1]
    final = ''
    seg_list = jieba.posseg.cut(lines)
    for seg in seg_list:
        if seg.flag in alist:
            final = final + " " + seg.word
    seg_list = final

    seg_list = ' '.join(re.findall(u'[\u4e00-\u9fff]+', seg_list))  ############去标点符号
    seg_list = seg_list.split()

    final = ''

    n = len(seg_list)
    for i in range(n):
        if seg_list[i] not in stopwords:  ##############去停用词
            final += seg_list[i]
            final += " "
    return final


def creat_patent_dic(csv_reader):
    patents = {}
    PatToSimPats = {}
    PatentClass = {}
    for line in csv_reader:
        apatent = []
        sim_patent = []
        patent_name = line[0]
        patent_name_list = split_word(patent_name)

        principal_claim = line[1]
        principal_claim_list = re.split(r'[,|.|;|，|。|；]', principal_claim)
        i = 0
        while i < len(principal_claim_list):
            principal_claim_list[i] = split_word(principal_claim_list[i])
            if len(principal_claim_list[i]) == 0:
                principal_claim_list.remove(principal_claim_list[i])
            else:
                i = i + 1

        #        function = line[2]   # 是否不要 或者提取function部分可以优化。
        #        function_list = re.split(r'[,|.|;|:|，|。|；|：]',function)
        #        j = 0
        #        while j < len(function_list):
        #            function_list[j] = split_word(function_list[j])
        #            if len(function_list[j]) == 0:
        #                function_list.remove(function_list[j])
        #            else:
        #                j = j+1
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
            patents[patent_name + " "] = apatent
            PatToSimPats[patent_name + " "] = sim_patent
            PatentClass[patent_name + " "] = MainClass
        elif patent_name + "  " not in patents:
            patents[patent_name + "  "] = apatent
            PatToSimPats[patent_name + "  "] = sim_patent
            PatentClass[patent_name + "  "] = MainClass
        elif patent_name + "   " not in patents:
            patents[patent_name + "   "] = apatent
            PatToSimPats[patent_name + "   "] = sim_patent
            PatentClass[patent_name + "   "] = MainClass
        elif patent_name + "    " not in patents:
            patents[patent_name + "    "] = apatent
            PatToSimPats[patent_name + "    "] = sim_patent
            PatentClass[patent_name + "    "] = MainClass
        elif patent_name + "     " not in patents:
            patents[patent_name + "     "] = apatent
            PatToSimPats[patent_name + "     "] = sim_patent
            PatentClass[patent_name + "     "] = MainClass
    return patents, PatToSimPats


def Patent_similarity(Apatents,Bpatents):   # 衡量专利之间的相似度，越小越好
    sim_dic = {}
    for patentA in Apatents:
        for patentB in Bpatents:
            patentA_list = Apatents[patentA]
            patentB_list = Bpatents[patentB]
            VectorA = vectors(patentA_list)
            VectorB = vectors(patentB_list)
#            VectorA = np.array(VectorA)
#            VectorB = np.array(VectorB)
#            dist = cosine_distance(VectorA, VectorB)
#            dist = cosine_similarity(VectorA, VectorB)
            dist = np.linalg.norm(VectorA - VectorB)     #向量之间距离，欧式
            if patentA not in sim_dic:
                dic = {}
                dic[patentB] = dist
                sim_dic[patentA] = dic
            else:
                dic = sim_dic[patentA]
                dic[patentB] = dist
                sim_dic[patentA] = dic

    return sim_dic


'''
def weight(n):
    #n=len(set_pre)
    w=[]
    if n==1:
        w=[1]

    if n==2:
        w=[0.5,0.5]

    if n>2:
        for i in range(n):
            if i+1<= n/2:
                w.append(n/2-(i+1)+1)
            if i+1>n/2:
                w.append((i+1)-n/2+1)
        #print(w)
        # sum=0
        # for j in range(len(w)):
        #     sum=sum+w[j]
        # #w=np.array(w)
        # for i in range(len(w)):
        #     w[i]=w[i]/sum
    return w
'''
def calculate(corpus):
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]  #######统计词频 并用词频表示产品向量 [(0, 1), (1, 1), (2, 1), (3, 1),.....]

    tfidf_vectors = model[doc_vectors]  #########计算每个词的tfidf 并用tfidf表示产品向量 [(0, 0.3073420004606649), (1, 0.175102025466),....]
    return tfidf_vectors



def vectors(sentence_alist):
    a = np.zeros(29002)                     #初始化
    # num = len(sentence_alist)
    # #print(num)
    # weight_vector = weight(num)
    i=0
    for sent in sentence_alist:
        sent = sent.split()
#        print(sent)
        tfidf_vectors= calculate([sent])
        sentence_vector = np.zeros(29002)                 # 维数，b为最后生成的句向量
        for val in tfidf_vectors[0]:
            # print(val[1])
            temp0 = val[0]
            temp1 = val[1]
            sentence_vector[temp0] = temp1

        a = a + sentence_vector
        i = i + 1
    return a

def TopKPatents(PatentToPatent_all,k):
    alist = []
    count = 0
    for key,val in PatentToPatent_all:

        if count==k:
            break
        count += 1
        alist.append(key)
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
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision,recall

f = open(u'weight_result.txt', "w+",encoding= 'utf-8')                 #保存去停用词，标点符号，空格后的数据

Apatents,APatToSimPats = creat_patent_dic(csv_reader1)
Bpatents,BPatToSimPats = creat_patent_dic(csv_reader2)

sim_dic = Patent_similarity(Apatents,Bpatents)
#print(sim_dic)                                  #[a:[bi:欧式距离],[bi2:distance2].....]



k = 0
precision_sum, recall_sum = 0,0
for patent in sim_dic:

    PatentToPatent_all = {}
    PatentToPatent = sim_dic[patent]
    for ToPatent in PatentToPatent:
        PatentToPatent_all[ToPatent] = PatentToPatent[ToPatent]



    PatentToPatent_all = sorted(PatentToPatent_all.items(),key = lambda PatentToPatent_all:PatentToPatent_all[1],reverse=True)
    #print(PatentToPatent_all)
    alist = TopKPatents(PatentToPatent_all,5)    # 推荐出来的相似专利的结果
    simPat = APatToSimPats[patent]            # 真实结果
    f.write(str(patent))
    f.write(str(alist)+"\n")
    print(patent,alist)  # 推荐结果
    f.write(str(patent))
    f.write(str(simPat)+"\n")
    print(patent,simPat) # 真实结果
    precision, recall = Recommend_Effect(alist, simPat)
    f.write(str(precision))
    f.write(str(recall)+"\n")
    print(precision, recall)
    f.write("-------------------------------------------------"+"\n")
    print("---------------------------------------------------")
    k = k + 1
    precision_sum = precision_sum + precision
    recall_sum = recall_sum + recall
#print(precision_sum,recall_sum)
#print(k)
f.write(str(precision_sum/k))
f.write(str(recall_sum/k)+"\n")
print(precision_sum/k)
#F1 = (k/precision_sum) + (k/recall_sum)
#print(1/F1)

f.close()