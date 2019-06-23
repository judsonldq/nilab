# -*- coding: utf-8 -*-
import logging
from gensim.models import word2vec
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence('专利数据.txt')
    model = word2vec.Word2Vec(sentences, size=350,sample=2e-3,negative=5,sg=1,alpha=0.005,min_alpha=0.00005,window=5,iter=15,min_count=1,workers=4)

    #保存模型，供日後使用
    model.save("word2vec_all.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
    # print(model.wv['水下机器人'])
