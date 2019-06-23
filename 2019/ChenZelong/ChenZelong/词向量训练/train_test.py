# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec
from gensim import models
import logging

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence('专利数据.txt')
#    model = models.Word2Vec.load('F:\\基于异质信息网络的专利分析\\词向量训练\\word2vec_all.model')
    lines = open(u'test.txt', "r+", encoding='utf-8')
    alist = []
    score_list = []
    parameter_list = []
    for line in lines:
        line = line.split()
        if "\ufeff" in line[0]:
            line[0] = line[0][1:]
        alist.append(line)
    parameter = []
    sample_alist = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01]
    alpha_alist = [0.005,0.01,0.02,0.025,0.05]

    for sample in sample_alist:
        for alpha in alpha_alist:
            model = word2vec.Word2Vec(sentences, size=340,sample=sample,negative=5,sg=1,alpha=alpha,window=5,iter=15,min_count=1,workers=4)
            scores = 0
            print("輸入兩個詞，則去計算兩個詞的餘弦相似度")
            for word in alist:
                score = model.similarity(word[0], word[1])
                scores = scores + score
            score_list.append(scores)
            parameter_list.append([sample,alpha])
            print(sample,alpha)
            print(scores)
    max_score = max(score_list)
    idx = score_list.index(max_score)
    print(parameter_list[idx])
                #保存模型，供日後使用
#                model.save("F:\\基于异质信息网络的专利分析\\词向量训练\\word2vec_all.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
