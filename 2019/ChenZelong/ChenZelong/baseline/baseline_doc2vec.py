import gensim
import csv
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import linecache


TaggededDocument = gensim.models.doc2vec.TaggedDocument
f = open(u'result_bldoc.txt', "w+",encoding= 'utf-8')


def get_datasest():
    with open('output2.txt', 'r', encoding='utf8') as cf:
        docs = cf.readlines()
        print(len(docs))
    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def train(x_train, size=200, epoch_num=5):
    model_dm = Doc2Vec(x_train, min_count=5, window=10, size = size, sample=1e-4,  workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('doc2vec_1.model')
    return model_dm
'''
def test(test_text):
    inferred_vector_dm = model_dm.infer_vector(test_text,steps = 0, alpha=0.025)
    sims = model_dm.docvecs.most_similar(positive=[inferred_vector_dm], topn=5)
    return sims
'''
def recommend_effect(alist,simPat):
    FP, FN, TP = 0, 0, 0
    for p1 in alist:
        #p1 = p1.strip()
        if p1 in simPat:
            TP = TP + simPat.count(p1)
        elif p1 not in simPat:
            FP = FP + 1
    for p2 in simPat:
        if p2 not in alist:
            FN = FN + 1
    precision = TP/(TP+FP)
    #recall = TP/(TP+FN)
    return precision


if __name__ == '__main__':
    x_train = get_datasest()
    model_dm = train(x_train)
    model_dm = Doc2Vec.load('doc2vec_1.model')
    num = 0
    n=0
    average=0
    for line in open('output1.txt', 'r', encoding='utf-8'):
        n=n+1
        get_out = []
#        real_out = []
        num += 1
#        count2 = linecache.getline('/Users/luw/Desktop/毕设/real_sim.txt', num).replace("\n", "")
#       real_out.append(str(count2))
        line = str(line).strip()
        line = line.split(' ')
        inferred_vector_dm = model_dm.infer_vector(list(line), steps=0, alpha=0.025)    #词向量
        print(inferred_vector_dm)

        #sims = test(list(line))
        sims = model_dm.docvecs.most_similar(positive=[inferred_vector_dm], topn=3)
        f.write('测试集专利编号 ' + str(num) + ' 对比集中相似度高的前5个专利编号')
        for count, sim in sims:
            f.write(' ' + str(count + 1) + ' ')
            get_out.append(count+1)

        get_count = linecache.getline('result4.txt', n).replace("\n", "")
        a = ['[', ']']
        for aa in a:
            get_count = get_count.replace(aa, ' ')

        li = get_count.rsplit(',')
        li_change = []
        for m in range(len(li)):
            li[m] = li[m].replace(' ', '')
            if li[m] != '':
                li_change.append(int(li[m]))
        precision=recommend_effect(get_out,li_change)
        average = average + precision
        f.write("准确率"+str(precision))

        f.write('\n')
    average = average / 529
    print(average)

    f.close()

