import csv
import jieba
import jieba.posseg as pseg 
import re


csv_reader = csv.reader(open('F:\\基于异质信息网络的专利分析 - new\\词向量训练\\专利-机制-功能-主分类号-分类号-相似专利.csv','r',encoding='gb18030'))      ##原始数据
# jieba.load_userdict("F:\\基于异质信息网络的专利分析\\杂七杂八的数据\\自定义词典.txt")                     #加载自定义词典
f = open(u'F:\\基于异质信息网络的专利分析 - new\\新词发现\\发现新词去重后的.txt', "w+",encoding= 'utf-8')                 #保存去停用词，标点符号，空格后的数据

stopwords = {}.fromkeys([line.rstrip() for line in open("F:\\基于异质信息网络的专利分析 - new\\新词发现\\动词表.txt",'r', encoding='UTF-8')])     #停用词路径

# alist = ["n","vn","v","j","nz","l","s","ng","f","nr","m","vg","nz","ng","x","q","d","eng","ns","nr"]
alist = ["n","vn","v","j","nz","l","s","ng","f","nr","m","vg","ng","q","d","eng","an","ad","a","d","vd","z","ns","j"]
db = "[！ ？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]"


NER = {}
for line in csv_reader:
    line = str(line[0]+"。"+line[1]+line[2]).replace('\n','')
#    line = str(line[0]).replace('\n','')
    seg = jieba.posseg.cut(line)
    l_word = []
    seq = ""
    for i in seg:
#        print(i.word,i.flag)
        if i.flag == "an":
            i.flag = "n"
        if i.flag == "vd":
            i.flag = "v"
        if i.flag == "ad":
            i.flag = "a"
        if i.flag == "vg":
            i.flag = "v"
        if i.flag == "nz":
            i.flag = "1"
        if i.flag == "vn":
            i.flag = "2"
        if i.flag == "zg":
            i.flag = "3"
        if i.flag == "ng":
            i.flag = "4"
        if i.flag == "eng":
            i.flag = "5"
        if i.flag == "nr":
            i.flag = "6"
        if i.flag == "ns":
            i.flag = "7"         
        if i.flag not in alist and len(i.flag) >= 2:
            i.flag = "0"
        seq = seq + i.flag
        l_word.append(i.word)
#    print(seq,l_word)


    m = [(m.start(),m.end()) for m in re.finditer("5?kngq|ngnq|ngq|ng|nq|5k5q|5?k?n{2,}1?|[n]+a|([n]+)q{1}|[n]+|[n]{2,}1?|snn|nsn|n?snq?|[n]+s|1n|n2n|n2|an|4n|5q|a4|n6|aq|vq|zv|la|a?3?|qn|b7|52n|2n|n4|5l|a?nk|fnv|nv|av|nq|b1|7b|31|bn", seq)]

 

    m = sorted(m,key=(lambda x:x[0]))
    m.reverse()

    for tul in m :
        res = "".join(l_word[tul[0]:tul[1]])
#        print(res)
        for stopword in stopwords:
            if stopword in res and (res.index(stopword) == 0 or res.index(stopword) + len(stopword) == len(res)) and (stopword == l_word[tul[0]] or stopword == l_word[tul[1]-1]):
                idx = res.index(stopword)
                if idx != 0:  
                    res = res[0:idx]
                if idx == 0:
                    res = res[len(stopword):]                    
        res = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；“”，《》<>：:{}[]|\~!?。？、~@#￥%……&*（）]+", "",res)  ############去标点符号
        res = re.sub(db, "", res)
#        print(res)

        pattern = re.compile(r'[A-Za-z0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*[A-Za-z0-9u0060u0021-u002cu002e-u002fu003a-u003fu2200-u22ffuFB00-uFFFDu2E80-u33FF]*$')  
        match = pattern.match(res)
        if match:
            flag = False
        else:
            flag = True 
#        print(res)
        res = re.split('[0-9|A-Z|a-z]+$',res)[0] 
        res = re.split('^[0-9]+',res)[-1]
        res = re.split('的$|了$|该$|来$|所述$|以$|是$|在$|在于$|与$|将$|为$|。$|本发明$|防止$|及$|共$|加入$|外$|使$|包括$|并且$|至少$|还$|不同$|对$|和$|现有$|及其$|其中$|分别$|其$|以$|第一$|第二$|第三$|第四$|第五$|两个$|一个$|三个$|预$|而$|可$|有$',res)[0]
        res = re.split('^S1|^S2|^S3|^S4|^S5|^S6|^S7|^S8|^S9|^nm|^ml|mm|^个|^是|^所述|^了|^该|^来|^在|^在于|^。|^的|^与|^将|^为|^倍|^及|^使|^用|^包括|^并且|^至少|^还|^不同|^对|^和|^现有|^该|^及其|^之间|^一种|^其中|^分别+',res)
        if len(res) == 2:
            res = res[1]
        else: 
            res = res[0]
        if res not in NER and len(res) > 1 and flag:
            NER[res] = 1
        if res in NER and len(res) > 1 and flag:
            NER[res] += 1
#    print(res)

for val in NER:
    if NER[val] >= 2:
        f.write(val + '\n')

f.close()