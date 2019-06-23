"""
# randomly sampling 245 beats form all the beats[75,75,75,13,7]
def sampleRandomly(path):
    listOfFolder = os.listdir(path)
    samplePath = '/home/lab307/chen/data_process/mean/common/'
    # 6602,105,264,15,7
    classesDict = {'0': 75, '1': 75, '2': 75, '3': 13, '4': 7}
    classesList = [[], [], [], [], []]
    for folder in listOfFolder:
        folderPath = os.path.join(path, folder)
        listOfFile = os.listdir(folderPath)
        for File in listOfFile:
            label = (File.split('.')[0]).split('_')[-1]
            classesList[int(label)].append(File)
    for i in range(len(classesList)):
        print(len(classesList[i]))
    for k in range(len(classesList)):
        sampleList = random.sample(classesList[k], classesDict[str(k)])
        for j in range(len(sampleList)):
            folder = sampleList[j].split('_')[0]
            readPath = os.path.join(path, folder, sampleList[j])
            writePath = os.path.join(samplePath, sampleList[j])
            f = open(readPath, 'r')
            g = open(writePath, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()

    print('Sampling is done!')


path = '/home/lab307/chen/data_process/mean/first/'
sampleRandomly(path)


# In[ ]:


# randomly sampling 245 beats form all the beats[75,75,75,13,7]
def sampleRandomly(path):
    listOfFile = os.listdir(path)
    samplePath = '/home/lab307/chen/data_process/mean/sample/'
    # 6602,105,264,15,7
    classesDict = {'0': 400, '1': 100, '2': 200, '3': 15, '4': 7}
    classesList = [[], [], [], [], []]
    for File in listOfFile:
        label = (File.split('.')[0]).split('_')[-1]
        classesList[int(label)].append(File)
    for i in range(len(classesList)):
        print(len(classesList[i]))
    for k in range(len(classesList)):
        sampleList = random.sample(classesList[k], classesDict[str(k)])
        for j in range(len(sampleList)):
            folder = sampleList[j].split('_')[0]
            readPath = os.path.join(path, sampleList[j])
            writePath = os.path.join(samplePath, sampleList[j])
            f = open(readPath, 'r')
            g = open(writePath, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()

    print('Sampling is done!')


path = '/home/lab307/chen/data_process/2D/train/'
sampleRandomly(path)


# In[ ]:



def computeToRelative(readPath, writePath, classSamples):
    fileList = os.listdir(readPath)
    count = 0
    # get data_process and label from 1D file
    for File in fileList:
        label = (File.split('.')[0]).split('_')[-1]
        readFilePath = os.path.join(readPath, File)
        f = open(readFilePath, 'r')
        chipBeat = []
        length = 0
        for lines in f.readlines():
            temp = lines.strip().split(',')
            chipBeat.append([float(temp[0]), float(temp[1])])
            length += 1
        # initial of transform
        lengthList = list(range(300))
        for i in range(classSamples[int(label)]):
            index = random.sample(lengthList, 1)[0]
            lengthList.remove(index)
            everyBeat = []
            for j in range(len(chipBeat)):
                angle = round(chipBeat[j][1] - chipBeat[index][1], 5)
                distance = round(chipBeat[j][0] * chipBeat[index][0] * math.cos(angle), 5)
                everyBeat.append([str(distance), str(angle)])
            fileName = str(count) + '_' + label + '.csv'
            count += 1
            writeFilePath = os.path.join(writePath, fileName)
            g = open(writeFilePath, 'a')
            for lines in everyBeat:
                g.write(','.join(lines) + '\n')
            g.close()


# In[ ]:


readPath = '/home/lab307/chen/data_process/2D/sample/'
writePath = '/home/lab307/chen/data_process/2D/relative/'
# {'0':400,'1':100,'2':200,'3':15,'4':7}
classSamples = [4, 5, 4, 10, 10]
computeToRelative(readPath, writePath, classSamples)

# In[ ]:


readPath = '/home/lab307/chen/data_process/2D/test/'
writePath = '/home/lab307/chen/data_process/2D/relative_test/'
classSamples = [1, 1, 1, 1, 1]
computeToRelative(readPath, writePath, classSamples)

# In[ ]:


readPath = '/home/lab307/chen/data_process/2D/val/'
writePath = '/home/lab307/chen/data_process/2D/relative_val/'
classSamples = [1, 1, 1, 1, 1]
computeToRelative(readPath, writePath, classSamples)


# In[5]:


# find the left twenty five minutes beats and copy it into twenty folders
def lastHalfData():
    allDataPath = '/home/lab307/chen/data_process/mean/first/'
    fiveDataPath = '/home/lab307/chen/data_process/mean/5minutes/'
    twentyDataPath = '/home/lab307/chen/data_process/mean/25minutes/'
    allDataFolderList = os.listdir(allDataPath)
    fiveDataFolderList = os.listdir(fiveDataPath)
    for folder in allDataFolderList:
        flag = 0
        allDataFolderPath = os.path.join(allDataPath, folder)
        fiveDataFolderPath = os.path.join(fiveDataPath, folder)
        fiveDataFileList = os.listdir(fiveDataFolderPath)
        allDataFileList = os.listdir(allDataFolderPath)
        # find biggest num in fiveDataFile
        for File in fiveDataFileList:
            num = int(File.split('_')[1])
            if flag < num:
                flag = num
        twentyDataFolderList = os.listdir(twentyDataPath)
        twentyDataFolderPath = os.path.join(twentyDataPath, folder)
        if folder not in twentyDataFolderList:
            os.mkdir(twentyDataFolderPath)
        # the num bigger than the biggest num above,the file is belong to twenty
        for File in allDataFileList:
            num = int(File.split('_')[1])
            if num > flag:
                oldPath = os.path.join(allDataFolderPath, File)
                newPath = os.path.join(twentyDataFolderPath, File)
                f = open(oldPath, 'r')
                g = open(newPath, 'w')
                strings = f.read()
                f.close()
                g.write(strings)
                g.close()
    print('the last twenty five minutes beats are chosed over!')


lastHalfData()

# In[14]:


# count different class nums
import os


def countNum(path, classes):
    listOfFile = os.listdir(path)
    classesList = []
    for i in range(classes):
        classesList.append([])
    for File in listOfFile:
        label = (File.split('.')[0]).split('_')[-1]
        classesList[int(label)].append(File)
    for i in range(len(classesList)):
        print(len(classesList[i]))


path = '/home/lab307/chen/data_process/mean/train/'
countNum(path, 5)


# 87899,2670,6995,802,15
# 72358,8055,7241,149,6889,802,83,2436,2,106,229,0,15,16,0


# In[ ]:


# [6602,105,264,15,7]


# In[ ]:


# 从多类标签转换为5类标签，按标签存档
def build_label(path1, path2, classesList):
    folderList = os.listdir(path1)
    count = 0
    for folder in folderList:
        folderpath = os.path.join(path1, folder)
        fileList = os.listdir(folderpath)
        for files in fileList:
            labels = (files.split('.')[0]).split('_')[-1]
            target = 0
            if labels in classesList[0]:
                target = '0'
            elif labels in classesList[1]:
                target = '1'
            elif labels in classesList[2]:
                target = '2'
            elif labels in classesList[3]:
                target = '3'
            elif labels in classesList[4]:
                target = '4'
            if target != 0:
                newfolder = os.listdir(path2)
                writepath = os.path.join(path2, target)
                if target not in newfolder:
                    os.makedirs(writepath)
                oldname = os.path.join(folderpath, files)
                newfile = str(count) + '_' + labels + '_' + target + '.csv'
                newname = os.path.join(writepath, newfile)
                f = open(oldname, 'r')
                g = open(newname, 'a')
                for lines in f.readlines():
                    g.write(lines)
                f.close()
                g.close()
                count += 1


# In[ ]:


path1 = '/home/lab307/chen/data_process/same/20_25/'
path2 = '/home/lab307/chen/data_process/same/20_25_T/'
classes_list = [['1', '2', '3', '11', '34'], ['4', '7', '8', '9'], ['5', '10'], ['6'], ['12', '13', '38']]
build_label(path1, path2, classes_list)


# In[ ]:


def count_classes_num(path):
    folderlist = os.listdir(path)
    countlist = []
    for folder in folderlist:
        countlist.append(len(os.listdir(os.path.join(path, folder))))
    print(countlist, sum(countlist))
    return countlist


path1 = '/home/lab307/chen/data_process/5/12_T/'
train_countlist = count_classes_num(path1)


# In[ ]:


# 从用户5类标签存成以5类标签为文件夹
def split_class(path1, path2):
    folderlist = os.listdir(path1)
    # basepath = '/home/dl307/chen/data_process/5/'
    count = 0
    for folder in folderlist:
        folderpath = os.path.join(path1, folder)
        filelist = os.listdir(folderpath)
        for files in filelist:
            label = (files.split('.')[0]).split('_')[-1]
            target = (files.split('.')[0]).split('_')[-2]
            newname = str(count) + '_' + target + '_' + label + '.csv'
            newpath = os.path.join(path2, label, newname)
            oldpath = os.path.join(folderpath, files)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            count += 1
            if count % 5000 == 0: print('finish %s!' % count)


# In[ ]:


path1 = '/home/lab307/chen/data_process/5/20_P/'
path2 = '/home/lab307/chen/data_process/5/20_T/'
split_class(path1, path2)


# In[ ]:


def count_folder_lines(path, length):
    folderlist = os.listdir(path)
    for folder in folderlist:
        folderpath = os.path.join(path, folder)
        filelist = os.listdir(folderpath)
        for files in filelist:
            count = 0
            filepath = os.path.join(path, folder, files)
            f = open(filepath, 'r')
            for lines in f.readlines():
                count += 1
            if count != length: print(filepath, count)


def count_lines(path, length):
    filelist = os.listdir(path)
    for files in filelist:
        count = 0
        filepath = os.path.join(path, files)
        f = open(filepath, 'r')
        for lines in f.readlines():
            count += 1
        if count != length: print(filepath, count)


# In[ ]:


length = 64
# path = '/home/dl307/chen/data_process/1D25_5/'
# path = '/home/dl307/chen/data_process/5/'
# path = '/home/lab307/chen/data_process/testdata/hour/'
# path = '/home/lab307/chen/data_process/5/20_raw/'
count_folder_lines(path, length)


# count_lines(path,length)


# In[11]:


# 前5分钟作为train
# 后25分钟作为test
# 后24个记录划分到test
def trans_train(path, mode=0):
    # train_path = '/home/lab307/chen/data_process/mean/train/'
    train_path = '/home/lab307/chen/data_process/mean/test/'
    folderlist = os.listdir(path)
    if mode == 0:
        # count=0
        # count = 245
        count = 34214
        for folder in folderlist:
            folderpath = os.path.join(path, folder)
            filelist = os.listdir(folderpath)
            for files in filelist:
                label = (files.split('.')[0]).split('_')[-1]
                oldpath = os.path.join(folderpath, files)
                newname = str(count) + '_' + label + '.csv'
                newpath = os.path.join(train_path, newname)
                f = open(oldpath, 'r')
                g = open(newpath, 'a')
                strings = f.read()
                f.close()
                g.write(strings)
                g.close()
                count += 1
                if count % 5000 == 0: print('finish %s!' % count)
    if mode == 1:
        count = 0
        for files in folderlist:
            label = (files.split('.')[0]).split('_')[-1]
            oldpath = os.path.join(path, files)
            newname = str(count) + '_' + label + '.csv'
            newpath = os.path.join(train_path, newname)
            f = open(oldpath, 'r')
            g = open(newpath, 'w')
            strings = f.read()
            f.close()
            g.write(strings)
            g.close()
            count += 1
    print(count)


# In[12]:


# path2 = '/home/lab307/chen/data_process/mean/5minutes/'
# path2 = '/home/lab307/chen/data_process/mean/25minutes/'
path2 = '/home/lab307/chen/data_process/mean/second/'
# path2 = '/home/lab307/chen/data_process/mean/common/'
trans_train(path2, mode=0)


# In[ ]:


# count the number of different labels
def countLabel(path):
    fileList = os.listdir(path)
    count = [0, 0, 0, 0, 0]
    for File in fileList:
        label = (File.split('.')[0]).split('_')[-1]
        count[int(label)] += 1
    print(count)
    # [6602, 105, 264, 15, 7]


# In[ ]:


path = '/home/lab307/chen/data_process/fft/all/'
# path = '/home/lab307/chen/data_process/pole/train/'
countLabel(path)


# In[15]:


# randomly sampling from all data_process folder
def sampleRandomly(path, sampleNumList):
    fileList = os.listdir(path)
    trainPath = '/home/lab307/chen/data_process/5Classes/train/'
    testPath = '/home/lab307/chen/data_process/5Classes/test/'
    classesList = [[], [], [], [], []]
    for File in fileList:
        label = File.split('.')[0].split('_')[-1]
        classesList[int(label)].append(File)
    for k in range(len(classesList)):
        sampleList = random.sample(classesList[k], sampleNumList[k])
        for name in classesList[k]:
            readPath = os.path.join(path, name)
            f = open(readPath, 'r')
            if name not in sampleList:
                writePath = os.path.join(testPath, name)
            else:
                writePath = os.path.join(trainPath, name)
            g = open(writePath, 'w')
            text = f.read()
            f.close()
            g.write(text)
            g.close()


# In[ ]:


# [87899, 2670, 6995, 802, 15]
# [6602,105,264,15,7]
sampleNumList = [200, 100, 150, 15, 7]
path = '/home/lab307/chen/data_process/5ClassesDataAll/all/'
sampleRandomly(path, sampleNumList)


# In[13]:


# move to val
def move_num(path1):
    filelist = os.listdir(path1)
    vallist = random.sample(filelist, 5000)
    valpath = '/home/lab307/chen/data_process/mean/val/'
    for files in vallist:
        oldpath = os.path.join(path1, files)
        newpath = os.path.join(valpath, files)
        os.rename(oldpath, newpath)


path1 = '/home/lab307/chen/data_process/mean/test/'
move_num(path1)


# In[ ]:


# tansfrom timeData to poleSystem
# classSamples element is the number of expandation of each beat
def transfromTo2D(readPath, writePath, classSamples):
    fileList = os.listdir(readPath)
    count = 0
    # get data_process and label from 1D file
    for File in fileList:
        label = (File.split('.')[0]).split('_')[-1]
        readFilePath = os.path.join(readPath, File)
        f = open(readFilePath, 'r')
        chipBeat = []
        length = 0
        for lines in f.readlines():
            chipBeat.append(float(lines))
            length += 1
        # initial of transform
        maxValue = round(max(chipBeat), 5)
        minValue = round(min(chipBeat), 5)
        interval = round((maxValue - minValue) / 2, 5)
        timeStep = round(1.0 / (2 * length), 5)
        lengthList = list(range(250))
        # transform data_process and expand samples
        # [6602, 105, 264, 15, 7]
        for i in range(classSamples[int(label)]):
            index = random.sample(lengthList, 1)[0]
            lengthList.remove(index)
            indexValue = round((chipBeat[index] - minValue - interval) / (maxValue - minValue), 5)
            everyBeat = []
            for j in range(len(chipBeat)):
                if j != index:
                    theta = round(math.atan((chipBeat[j] - chipBeat[index]) / (timeStep * (index - j))) * math.pi / 180,
                                  5)
                else:
                    theta = 0
                distance = round(math.sqrt((chipBeat[index] - chipBeat[j]) ** 2 + (timeStep * (index - j)) ** 2), 5)
                everyBeat.append([float(distance), str(theta)])  # make2DDataIntoList
            fileName = str(count) + '_' + label + '.csv'
            count += 1
            writeFilePath = os.path.join(writePath, fileName)
            g = open(writeFilePath, 'a')
            for lines in everyBeat:
                lines[0] = str(lines[0])
                lines[1] = str(lines[1])
                g.write(','.join(lines) + '\n')
            g.close()


# In[ ]:


# [6602, 105, 264, 15, 7][100,80,90,60,8]
classSamples = [1, 2, 1, 4, 6]  # [6602,2625,3168,1200,980]
readPath = '/home/lab307/chen/data_process/mean/train/'
writePath = '/home/lab307/chen/data_process/pole/train/'
transfromTo2D(readPath, writePath, classSamples)

# In[ ]:


classSamples = [1, 1, 1, 1, 1]
readPath = '/home/lab307/chen/data_process/mean/val/'
writePath = '/home/lab307/chen/data_process/pole/val/'
transfromTo2D(readPath, writePath, classSamples)

# In[ ]:


classSamples = [1, 1, 1, 1, 1]
readPath = '/home/lab307/chen/data_process/mean/test/'
writePath = '/home/lab307/chen/data_process/pole/test/'
transfromTo2D(readPath, writePath, classSamples)


# In[ ]:


def transfromTo2D(readPath):
    fileList = os.listdir(path)
    count = 0
    # get data_process and label from 1D file
    for File in fileList:
        label = (File.split('.')[0]).split('_')[-1]
        readFilePath = os.path.join(readPath, File)
        f = open(readFilePath, 'r')
        chipBeat = []
        length = 0
        for lines in f.readlines():
            chipBeat.append(float(lines))
            length += 1
    return chipBeat


readPath = '/home/lab307/chen/data_process/mean/train/'
type(transfromTo2D(readPath)[0])

# In[ ]:


for thing in lista:
    print(thing)


# In[ ]:


def checkRaw(path, mode):
    folderList = os.listdir(path)
    if mode == 1:
        for folder in folderList:
            folderPath = os.path.join(path, folder)
            fileList = os.listdir(folderPath)
            for File in fileList:
                filePath = os.path.join(folderPath, File)
                f = open(filePath, 'r')
                for lines in f.readlines():
                    temp = lines.strip().split(',')
                    if len(temp) == 1:
                        print(folder, File)
                    break
    elif mode == 0:
        for File in folderList:
            filePath = os.path.join(path, File)
            f = open(filePath, 'r')
            for lines in f.readlines():
                temp = lines.strip().split(',')
                if len(temp) == 1:
                    print(File)
                break


# In[ ]:


path = '/home/lab307/chen/data_process/5ClassesDataAll/all/'
checkRaw(path, mode=1)


# In[ ]:


# 挑选common心跳

def sixth2five(old_path):
    new_path = '/home/lab307/chen/data_process/5/train_12/'
    # old_path = '/home/lab307/chen/data_process/5/val_12/'
    # sizelist = [75,75,75,13,7]
    # [3461, 103, 55, 1276, 39436]
    # trainlist = []
    # testlist = []
    # classes_list = ['1','2','3','4','5','6','7','8','9','10','11','13','34','38']
    classes_list = ['1', '2', '3', '11', '34', '4', '7', '8', '9', '5', '10', '6', '12', '13', '38']
    sixthlist = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    sixthlen = []
    fivelist = [[], [], [], [], []]
    fivelen = []
    transdict = {}
    trainlist = []
    # sizelist = [2500,800,700,66,4,100,40,1000,1,2000,80,200,0,10,0]#extend_all_7501
    # sizelist = [2500,800,700,66,4,40,20,800,1,2000,20,200,0,4,0]#all_7000+
    # sizelist = [2000,500,500,46,4,40,20,500,1,1500,20,100,0,4,0]#all_5000+
    # sizelist = [4000,1000,500,5,0,71,1,500,1,1000,53,500,0,8,0]
    # [20639, 3452, 1909, 10, 0, 142, 1, 602, 2, 3316, 106, 761, 0, 8, 0]
    # print(sum(sizelist))
    for i in range(len(classes_list)):
        transdict[classes_list[i]] = i
    # train_count = 6710
    # test_count = 59406
    filelist = os.listdir(old_path)
    for files in filelist:
        label = (files.split('.')[0]).split('_')[-2]
        target = (files.split('.')[0]).split('_')[-1]
        sixthlist[transdict[label]].append(files)
        fivelist[int(target)].append(files)
    for i in range(len(sixthlist)):
        sixthlen.append(len(sixthlist[i]))
    for i in range(len(fivelist)):
        fivelen.append(len(fivelist[i]))
        # for i in range(len(sixthlist)):
        # trainlist.extend(random.sample(sixthlist[i],sizelist[i]))
        # for i in range(len(trainlist)):
        # oldpath = os.path.join(old_path,trainlist[i])
        # newpath = os.path.join(new_path,trainlist[i])
        # os.rename(oldpath,newpath)

    print(sixthlen)
    print(fivelen)
    # extend_all
    # [74235, 8050, 7232, 229, 16, 345, 147, 6166, 2, 16423, 314, 857, 0, 43, 0]
    # [89762, 6660, 16737, 857, 43]
    # all
    # [74235, 8050, 7232, 229, 16, 149, 83, 2528, 2, 6883, 106, 801, 0, 15, 0]
    # [89762, 2762, 6989, 801, 15]
    # test
    # [69052, 7275, 6645, 229, 16, 148, 67, 2514, 2, 6692, 106, 799, 0, 15, 0]
    # [83217, 2731, 6798, 799, 15]
    # train
    # [5213, 769, 584, 0, 0, 1, 14, 414, 0, 263, 0, 102, 0, 50, 0]
    # [6566, 429, 263, 102, 50]



    # print ('train_count:%s  test_count:%s' % (train_count,test_count))
    # classes_list = [['1','2','3','11','34'],['4','7','8','9'],['5','10'],['6'],['12','13','38']]


# In[ ]:


path = '/home/lab307/chen/data_process/5/train_64/'
sixth2five(path)


# train[5200, 777,   589,   0,  0,   4, 26,   74, 0,  263,  0,  15, 0,  7, 0]
# test[65381, 6885, 6266, 223, 13, 139, 64, 2383, 2, 6338, 98, 760, 0, 15, 0]


# In[ ]:


# 挑选common心跳
# classes_list = [['1','2','3','11','34'],['4','7','8','9'],['5','10'],['6'],['13','38']]
def split_train_test(path):
    train_path = '/home/lab307/chen/data_process/5/train_12/'
    test_path = '/home/lab307/chen/data_process/5/test_12/'
    # sizelist = [75,400,75,100,50]
    # sizelist = [5000,700,2000,700,5]
    # [3422, 761, 8, 747, 26010]
    # trainlist = []
    # testlist = []
    train_count = 6710
    test_count = 69388
    sizedict = {'0': 0, '1': 100, '2': 50, '3': 100, '4': 5}
    folderlist = os.listdir(path)
    for folder in folderlist:
        folderpath = os.path.join(path, folder)
        filelist = os.listdir(folderpath)
        trainlist = random.sample(filelist, sizedict[folder])
        testlist = [terms for terms in filelist if terms not in trainlist]
        for files in trainlist:
            oldpath = os.path.join(folderpath, files)
            label = (files.split('.')[0]).split('_')[-1]
            target = (files.split('.')[0]).split('_')[-2]
            name = str(train_count) + '_' + target + '_' + label + '.csv'
            newpath = os.path.join(train_path, name)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            train_count += 1

        for files in testlist:
            oldpath = os.path.join(folderpath, files)
            label = (files.split('.')[0]).split('_')[-1]
            target = (files.split('.')[0]).split('_')[-2]
            name = str(test_count) + '_' + target + '_' + label + '.csv'
            newpath = os.path.join(test_path, name)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            test_count += 1
    print('train_count:%s  test_count:%s' % (train_count, test_count))


# In[ ]:


path1 = '/home/lab307/chen/data_process/5/12_T/'
split_train_test(path1)


# In[ ]:


def count_class(path):
    filelist = os.listdir(path)
    countlist = [0, 0, 0, 0, 0]
    for files in filelist:
        label = (files.split('.')[0]).split('_')[-1]
        countlist[int(label)] += 1
    print(countlist)


# In[ ]:


count_class('/home/lab307/chen/data_process/mean/train/')

# In[ ]:


count_class('/home/lab307/chen/data_process/5/train5/')

# In[ ]:


print(len(os.listdir('/home/lab307/chen/data_process/5/train5/')), len(os.listdir('/home/lab307/chen/data_process/5/test5/')))


# In[ ]:


def count_classes_num(path):
    folderlist = os.listdir(path)
    countlist = []
    for folder in folderlist:
        countlist.append(len(os.listdir(os.path.join(path, folder))))
    print(countlist)
    return countlist


path1 = '/home/lab307/chen/data_process/5/20_T/'
train_countlist = count_classes_num(path1)

# In[ ]:


path1 = '/home/lab307/chen/data_process/5/20_T/'
train_countlist = count_classes_num(path1)

# In[ ]:


path1 = '/home/dl307/chen/data_process/5/test5/'
test_countlist = count_classes_num(path1)

# In[ ]:


path1 = '/home/dl307/chen/data_process/5/val5/'
val_countlist = count_classes_num(path1)

# In[ ]:


sum_count = sum(train_countlist) + sum(test_countlist) + sum(val_countlist)
sum_count

# In[ ]:


for i in range(len(countlist)):
    print(5000 * (countlist[i] / sum(countlist)))

# In[ ]:


path1 = '/home/dl307/chen/data_process/train5/'
countlist = count_classes_num(path1)


# In[ ]:


# move to val
def move_num(path1):
    filelist = os.listdir(path1)
    vallist = random.sample(filelist, 5000)
    valpath = '/home/lab307/chen/data_process/mean/val/'
    for files in vallist:
        oldpath = os.path.join(path1, files)
        newpath = os.path.join(valpath, files)
        os.rename(oldpath, newpath)


path1 = '/home/lab307/chen/data_process/mean/test/'
move_num(path1)


# In[ ]:


# 构建一个均衡的train
def move_val(path):
    val_path = '/home/lab307/chen/data_process/5/val_64/'
    temp_path = '/home/lab307/chen/data_process/5/temp/'
    val_count = 6710
    trainlist = [[], [], [], [], []]
    sizelist = [75, 75, 75, 13, 7]
    vallist = [[], [], [], [], []]
    folderlist = os.listdir(path)
    for folder in folderlist:
        folderpath = os.path.join(path, folder)
        filelist = os.listdir(folderpath)
        for file in filelist:
            oldpath = os.path.join(folderpath, file)
            newpath = os.path.join(temp_path, file)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()

    newfilelist = os.listdir(temp_path)
    for file in newfilelist:
        label = (file.split('.')[0]).split('_')[-1]
        trainlist[int(label)].append(file)
    for i in range(5):
        vallist[i] = random.sample(trainlist[i], sizelist[i])
        for j in range(len(vallist[i])):
            oldpath = os.path.join(temp_path, vallist[i][j])
            namelist = vallist[i][j].split('_')
            newnamelist = [str(val_count), namelist[-2], namelist[-1]]
            newname = '_'.join(newnamelist)
            newpath = os.path.join(val_path, newname)
            val_count += 1
            os.rename(oldpath, newpath)
            f = open(oldpath,'r')
            g = open(newpath,'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            train_count += 1

            # print ('train_count:%s  test_count:%s' % (train_count,test_count))


# In[ ]:


path = '/home/lab307/chen/data_process/5/20_64_T/'
move_val(path)


# In[ ]:


def move_train_mini(path):
    train_path = '/home/dl307/chen/data_process/5/train5_mini/'
    train_count = 0
    filelist = os.listdir(path)
    trainlist = [[], [], [], [], []]
    minilist = [[], [], [], [], []]
    sizelist = [2000, 265, 600, 26, 14]
    for i in range(len(filelist)):
        label = (filelist[i].split('.')[0]).split('_')[-1]
        trainlist[int(label)].append(filelist[i])
        # print (len(trainlist[i]))
    for i in range(5):
        minilist[i] = random.sample(trainlist[i], sizelist[i])
        for j in range(len(minilist[i])):
            oldpath = os.path.join(path, minilist[i][j])
            label = (minilist[i][j].split('.')[0]).split('_')[-1]
            target = (minilist[i][j].split('.')[0]).split('_')[-2]
            name = str(train_count) + '_' + target + '_' + label + '.csv'
            newpath = os.path.join(train_path, name)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            train_count += 1
    print('train_count:%s' % (train_count))


# In[ ]:


path = '/home/dl307/chen/data_process/5/train5/'
move_train_mini(path)


# In[ ]:


# 把5类标签分离为train和test
def sample_train(path):
    train_path = '/home/lab307/chen/data_process/5/train5/'
    # test_path = '/home/lab307/chen/data_process/5/test5/'
    sizelist = [925, 29, 188, 0, 0]
    # [6491, 29, 188, 2, 0]
    trainlist = [[], [], [], [], []]
    train_count = 245
    filelist = os.listdir(path)
    for files in filelist:
        label = (files.split('.')[0]).split('_')[-1]
        trainlist[int(label)].append(files)
    for i in range(len(trainlist)):
        samplelist = random.sample(trainlist[i], sizelist[i])
        for files in samplelist:
            oldpath = os.path.join(path, files)
            label = (files.split('.')[0]).split('_')[-1]
            target = (files.split('.')[0]).split('_')[-2]
            newname = str(train_count) + '_' + target + '_' + label + '.csv'
            newpath = os.path.join(train_path, newname)
            f = open(oldpath, 'r')
            g = open(newpath, 'a')
            for lines in f.readlines():
                g.write(lines)
            f.close()
            g.close()
            train_count += 1
    print('train_count:%s' % (train_count))


# In[ ]:


path = '/home/lab307/chen/data_process/5/train5_raw/'
sample_train(path)


# In[ ]:


# 找‘2’的个数
def function1(path):
    count = 0
    folderlist = os.listdir(path)
    for folder in folderlist:
        folderpath = os.path.join(path, folder)
        filelist = os.listdir(folderpath)
        for files in filelist:
            target = (files.split('.')[0]).split('_')[-1]
            # if target == '5' or target == '10':
            if target == '13' or target == '38':
                count += 1
    print(count)


function1('/home/lab307/chen/data_process/5/20/')


# In[ ]:


def function2(path1, path2, classesList):
    fileList = os.listdir(path1)
    count = 0
    for files in fileList:
        labels = (files.split('.')[0]).split('_')[-1]
        target = 0
        if labels in classesList[0]:
            target = '0'
        elif labels in classesList[1]:
            target = '1'
        elif labels in classesList[2]:
            target = '2'
        elif labels in classesList[3]:
            target = '3'
        elif labels in classesList[4]:
            target = '4'
        if target != 0:
            newfolder = os.listdir(path2)
            if target not in newfolder:
                writepath = os.path.join(path2, target)
                os.makedirs(writepath)
            oldname = os.path.join(path1, files)
            newfile = str(count) + '_' + labels + '_' + target + '.csv'
            newname = os.path.join(writepath, newfile)
            count += 1
            f = open(oldname, 'r')
            g = open(newname, 'a')
            for lines in f.readlines():
                g.write(lines)
    f.close()
    g.close()


# In[ ]:


path1 = '/home/lab307/chen/data_process/5/20/100/'
path2 = '/home/lab307/chen/data_process/5/20_T/'
classes_list = [['1', '2', '3', '11', '34'], ['4', '7', '8', '9'], ['5', '10'], ['6'], ['13', '38']]
function2(path1, path2, classes_list)


# In[ ]:


def link_time_and_dwt(path1, path2):
    timeFileList = os.listdir(path1)
    dwtFileList = os.listdir(path2)
    for files in timeFileList:
        time_path = os.path.join(path1, files)
        dwt_path = os.path.join(path2, files)
        f = open(dwt_path, 'r')
        g = open(time_path, 'a')
        for lines in f.readlines():
            g.write(lines)
        f.close()
        g.close()


# In[ ]:


path1 = '/home/lab307/chen/data_process/5/test_64/'
path2 = '/home/lab307/chen/data_process/5/test_64_dwt/'
link_time_and_dwt(path1, path2)


# In[ ]:


def sep_time_series(path1, path2, length):
    mergeFileList = os.listdir(path1)
    count_length = length
    for files in mergeFileList:
        merge_path = os.path.join(path1, files)
        time_path = os.path.join(path2, files)
        f = open(merge_path, 'r')
        g = open(time_path, 'a')
        for lines in f.readlines():
            g.write(lines)
            count_length -= 1
            if count_length == 0:
                count_length = length
                break
        f.close()
        g.close()


# In[ ]:


path1 = '/home/lab307/chen/data_process/5/train_64/'
path2 = '/home/lab307/chen/data_process/5/test_copy/'
sep_time_series(path1, path2, 64)

# In[ ]:


from sklearn import preprocessing

# In[ ]:


lista = [2.0, 5.0, 1.2, -5.4, -7.4, -1.2]
normalizer = preprocessing.Normalizer()
normalizer.transform(lista)

# In[ ]:


import os
import numpy as np
import random
from random import shuffle
from sklearn.cross_validation import train_test_split


# In[ ]:


# [100,101,103,105,106,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124]
def getTime(path, patientList, timeFlag):
    fileList = os.listdir(path)
    trainPath = '/home/lab307/chen/data_process/5classes/train/'
    testPath = '/home/lab307/chen/data_process/5classes/test/'
    for File in fileList:
        patient = int(File.split('_')[0])
        time = int(File.split('_')[1])
        if patient in patientList and time <= 300:
            f = open(os.path.join(path, File))
            g = open(os.path.join(trainPath, File))


# In[ ]:


def checkRaw(path, mode):
    folderList = os.listdir(path)
    if mode == 1:
        for folder in folderList:
            folderPath = os.path.join(path, folder)
            fileList = os.listdir(folderPath)
            for File in fileList:
                filePath = os.path.join(folderPath, File)
                f = open(filePath, 'r')
                for lines in f.readlines():
                    temp = lines.strip().split(',')
                    if len(temp) == 1:
                        print(folder, File)
                    break
    elif mode == 0:
        for File in folderList:
            filePath = os.path.join(Path, File)
            f = open(filePath, 'r')
            for lines in f.readlines():
                temp = lines.strip().split(',')
                if len(temp) == 1:
                    print(folder, File)
                break


path = '/home/lab307/chen/data_process/mean/25minutes/'
checkRaw(path)


# In[ ]:


# tansfrom timeData to poleSystem
# classSamples element is the number of expandation of each beat
def transfromTo2D(readPath, writePath, classSamples):
    fileList = os.listdir(readPath)
    count = 0
    # get data_process and label from 1D file
    for File in fileList:
        label = (File.split('.')[0]).split('_')[-1]
        readFilePath = os.path.join(readPath, File)
        f = open(readFilePath, 'r')
        chipBeat = []
        length = 0
        for lines in f.readlines():
            chipBeat.append(float(lines))
            length += 1
        # initial of transform
        perPhase = round(2 * math.pi / 300, 5)
        lengthList = list(range(300))
        timeStep = round(1.0 / 360.0, 5)
        for i in range(classSamples[int(label)]):
            index = random.sample(lengthList, 1)[0]
            lengthList.remove(index)
            fileName = str(count) + '_' + label + '.csv'
            count += 1
            writeFilePath = os.path.join(writePath, fileName)
            g = open(writeFilePath, 'a')
            for j in range(len(chipBeat)):
                if j != index:
                    theta = round(math.atan((chipBeat[j] - chipBeat[index]) / (timeStep * (index - j))) * math.pi / 180,
                                  5)
                else:
                    theta = 0
                g.write(','.join([str(chipBeat[j]), str(theta)]) + '\n')
            g.close()


# In[ ]:


# [6602, 105, 264, 15, 7][100,80,90,60,8]
classSamples = [2, 60, 15, 90, 40]  # [6602,2625,3168,1200,980]
readPath = '/home/lab307/chen/data_process/mean/train/'
writePath = '/home/lab307/chen/data_process/pole/train/'
transfromTo2D(readPath, writePath, classSamples)

# In[ ]:


classSamples = [1, 1, 1, 1, 1]  # [6602,2625,3168,1200,980]
readPath = '/home/lab307/chen/data_process/mean/test/'
writePath = '/home/lab307/chen/data_process/pole/test/'
transfromTo2D(readPath, writePath, classSamples)

# In[ ]:


classSamples = [1, 1, 1, 1, 1]  # [6602,2625,3168,1200,980]
readPath = '/home/lab307/chen/data_process/mean/val/'
writePath = '/home/lab307/chen/data_process/pole/val/'
transfromTo2D(readPath, writePath, classSamples)


# In[ ]:


# compute chazhi
def chaZhi(readPath, ):
    os.lsitdir()
"""