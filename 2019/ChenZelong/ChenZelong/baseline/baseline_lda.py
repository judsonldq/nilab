from gensim import corpora, models, similarities
import linecache

f = open('output2.txt','r',encoding = 'utf8')
f1 = open('output1.txt','r',encoding = 'utf8')
f2 = open(u'result_lda.txt','w',encoding = 'utf8')


def recommend_effect(alist,simPat):
    FP, FN, TP = 0, 0, 0
    for p1 in alist:
        #p1 = p1.strip()
        if p1 in simPat:
            TP = TP + simPat.count(p1)
        elif p1 not in simPat:
            FP = FP + 1
    '''for p2 in simPat:
        if p2 not in alist:
            FN = FN + 1'''
    precision = TP/(TP+FP)
    return precision

lines = f.readlines()
corpus = []
for line in lines:
    line = line.split()
    corpus.append(line)
# print(corpus)

lines_ = f1.readlines()
query = []
for line in lines_:
    line = line.split()
    query.append(line)

dictionary = corpora.Dictionary(corpus)   # 得到单词的ID,统计单词出现的次数以及统计信息 ,建立词库表 ['具体情况']:0 ['内部']:1
texts = [dictionary.doc2bow(text) for text in corpus]   #######统计词频 并用词频表示产品向量 [(0, 1), (1, 1), (2, 1), (3, 1),.....]
#print(texts)
texts_tf_idf = models.TfidfModel(texts)[texts]       #########计算每个词的tfidf 并用tfidf表示产品向量 [(0, 0.3073420004606649), (1,0.175102025466),....]
#print(texts_tf_idf)

# 利用LDA做主题分类的情况
#print ("**************LDA*************")
lda = models.ldamodel.LdaModel(corpus=texts, id2word=dictionary, num_topics=600,update_every=0,passes=20)
texts_lda = lda[texts_tf_idf]
print (lda.print_topics(num_topics=600, num_words=5))

for doc1 in texts_lda:
    print (doc1)

index = similarities.MatrixSimilarity(texts_lda)


average=0

for i in range(len(query)):
    query_bow = dictionary.doc2bow(query[i])
    query_lda = lda[query_bow]
    print(query_lda)
    sims = index[query_lda]
    index_lda = []
    similar_lda = []
    f2.write('测试集专利编号 '+str(i+1)+ ' 对比集中相似专利编号（前5个） ')
    for j in range(5):
        index_lda.append(list(sims).index(max(sims)))
        similar_lda.append(max(sims))
        sims = list(sims)
        sims.remove(similar_lda[j])
    '''
    print("与输入集合第 %d 个产品最相似的是输出集合中第 %d(%f)、%d(%f) 、%d(%f)、%d(%f)、%d(%f)、%d(%f)、%d(%f)、%d(%f)、%d(%f)、%d(%f) 个产品"%((i+1),
                (int(index_lda[0])+1),float(similar_lda[0]),
                (int(index_lda[1])+1),float(similar_lda[1]),
                (int(index_lda[2])+1),float(similar_lda[2]),
                (int(index_lda[3])+1),float(similar_lda[3]),
                (int(index_lda[4])+1),float(similar_lda[4]),
                (int(index_lda[5])+1),float(similar_lda[5]),
                (int(index_lda[6])+1),float(similar_lda[6]),
                (int(index_lda[7])+1),float(similar_lda[7]),
                (int(index_lda[8])+1),float(similar_lda[8]),
                (int(index_lda[9])+1),float(similar_lda[9])
                ))
    '''
    get_count = linecache.getline('result4.txt', i+1).replace("\n", "")
    a = ['[', ']']
    for aa in a:
        get_count = get_count.replace(aa, ' ')

    li = get_count.rsplit(',')
    li_change = []
    for m in range(len(li)):
        li[m] = li[m].replace(' ', '')
        if li[m] != '':
            li_change.append(int(li[m]))

    precision = recommend_effect(index_lda, li_change)

    average = average + precision
    f2.write(str(index_lda[0]+1)+' '+str(index_lda[1]+1)+' '+str(index_lda[2]+1)+' '+str(index_lda[3]+1)+' '+str(index_lda[4]+1)+' ')
    f2.write("          准确率"+str(precision))
    f2.write('\n')
    #print(list(sorted_x[1]).index(max(sorted_x[1])))
average = average / 529
print(average)
f2.write(str(average))
f2.close()
