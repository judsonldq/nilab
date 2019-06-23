import  re
#import matplotlib.pyplot as plt
file=open("C:\\Users\wangwei\Desktop\毕业设计数据\sim_result_BUS_A_2018.01.26_11.26.26.txt")
#file=open("C:\\Users\wangwei\Desktop\sim_result.txt")
data=file.read().split( )
ylable=[]
for i in data:
    ylable.append(float(i)/128)

'''
print(min(ylable))    #最小带宽
print(max(ylable))
print((sum(ylable)/len(ylable)))   #平均带宽
'''
#plt.plot(xlable,ylable)
#plt.show()
#print(ylable)
#print('OK')

T=1800
bandWish=ylable
r=1     #播放速率为1M/s
w=r*60
#w=10
#print(bandWish)

#print(len(bandWish))

#贪婪传输找可优化段
rebuTime={}
play_t=0        #play_t为已经播放的时长，初始化为0
playData=0      #已经播放的数据量，初始化为0
VideoData=T*r
t=0            #t为下载时长,从第0秒开始下载
down_data=0        #已经下载的数据量，初始化为0
T_w=0           #t时刻缓存区的数据量，初始化为0
Data=[]         #t时刻已经下载的数据量,对应rebuTime{},即到达上限或者下限的数据量
Play_Data=[]    #t时刻已经播放的数据量，对应rebuTime{}
#print(bandWish)
re_cut={}
while down_data<=VideoData:                                #有可能下载完毕但是播放没有完毕
    if down_data==0 or down_data+bandWish[t]<playData+r:   #发生卡顿,只下载不播放
        rebuTime[t]="C"
        Data.append(down_data)
        Play_Data.append(playData)
        re_cut[t]="C"
        T_W=down_data-playData       #当前时刻缓存区内的数据
      #  print('发生卡顿', '卡顿时间:', t, ' ', '缓存区数据量', T_W,'已经下载的数据',down_data,'已经播放的',playData)
        while T_W+bandWish[t]<w:
            T_W=T_W+bandWish[t]
            down_data=down_data+bandWish[t]
       #     print('时间:', t, '带宽：', bandWish[t],'缓存区内带宽:',T_W)
            t=t+1
        #    print('下一秒的带宽为',bandWish[t])

        down_data=down_data+(w-T_W)
       # print('卡顿结束:','时间',t,'已经下载的',down_data,'已经播放的',playData)
        rebuTime[t]='E'
        Data.append(down_data)
        Play_Data.append(playData)
        t = t + 1

    else:                #不卡断，边下载边播放
       # print(t,'下一秒带宽为：',bandWish[t],'已经下载的:',down_data,'已经播放的',playData,'缓存区内剩余',down_data-playData)
        if down_data+bandWish[t]-playData-r<=w:
            down_data=down_data+bandWish[t]
        else:
          #  print('发生溢出',down_data-playData,w-(down_data-playData))
            rebuTime[t]='E'
            down_data=down_data+w-(down_data-playData)
            Data.append(down_data)
            Play_Data.append(playData)
            if len(rebuTime)!=len(Data):
                break
        play_t=play_t+1
        playData=playData+r
        t=t+1
    #if len(rebuTime)!=len(Data):
    #    print('wrong')
     #   print(len(rebuTime),len(Data),len(Play_Data))
#print(len(Data),len(rebuTime),len(Play_Data))
print('贪婪传输下载完毕','下载时间',t,'下载的数据量',down_data)


#print(rebuTime)
#print(Play_Data)
#print(Data)
#print(re_cut)
#print(len(rebuTime),len(Data),len(Play_Data))
#print(Data)
#可优化段
N=[[11,1754]]                       #可优化段
N_Data=[[60,1800]]                 #可优化段时刻对应的已经下载完成的数据量
N_PlayData=[[0,1742]]                                  #可优化段对应的已经播放的数据量
###############EDA下载
tail_P = 1000
connet_p = 1560
promotion_p = 1360
tail_time=10
promotion_time=0.5

#

X_D=[]
#X_T=[]
X_Time=[]        #最优的下载任务
X_Data=[]        #最优的下载任务对应的已经下载的数据量范围
##最大能耗
def bestRatio(start_t,end_t,start_d,u,v):                  #待测试
    print('   ')
    print('开始时',start_t,end_t,start_d)
    bsData=max(start_d,v[start_t])                                 #开始时刻的数据量 max(开始时刻缓存下限，开始时刻已经下载的数据量)
    #print('bsData',bsData)
    beData=bsData                #开始时刻的数据量
    for i in range(start_t,end_t):
        bsData=bsData+bandWish[i]                  #一会判断一下时start_t还是start_t-1
        #print (i,'时刻的信道',bandWish[i],'下载的数据量',bsData,"上限",u[i],'下限',v[i])
        if bsData>=u[i] or bsData>=1800:
            ActallyEnd_time=i+1
            bsData=u[i]
            break
        elif bsData<v[i] or i==end_t-1:
            #break
            print('发生卡顿,下载停止')
            return -1,-1,-1,-1,-1
    ####计算KI
    print(ActallyEnd_time)
    pre = [start_t, ActallyEnd_time]
    X_T=X_Time[:]
   # print('X_TIME',X_Time,'X_T',X_T)
    X_T.append(pre)
    c = sorted(X_T)
   # print(c)
    num = c.index(pre)
    pre_time = 100
    pro_time = 100
    if num >= 1:
        pre_t = c[num - 1][1]
        pre_time = c[num][0] - c[num - 1][1]
    if len(c) > num + 1:
        pro_t = c[num + 1][0]
        pro_time = c[num + 1][0] - c[num][1]
    if pre_time > tail_time and pro_time > tail_time:
        ki = pre[1] - pre[0] + tail_time
    elif pre_time > tail_time and pro_time <= tail_time:
        ki = pro_t - pre[0]
    elif pre_time <= tail_time and pro_time > tail_time:
        ki = pre[1] - pre_t
    elif pro_time <= tail_time and pro_time <= tail_time:
        ki = max(1, pro_t - pre_t - tail_time)

    if ki==0:
        print("出现错误")
        print(start_t,ActallyEnd_time)
        print(c)
        exit()
    else:
        print('开始时的数据量:',beData,'结束时的数据量',bsData,'消耗的总时间',ki)
        ni=(bsData-beData)/ki
    print('开始时间',start_t,'结束时间',ActallyEnd_time,'开始时的数据量:',beData,'结束时的数据量',bsData,'能耗比',ni)
   # print(ni)
    return start_t,ActallyEnd_time,ni,beData,bsData                               #####测试,返回下载的数据量段范围



index = 0
for i in N_Data:
    v = N_Data[index][1]-N_Data[index][0]
    Down_Time=N[index]                          #需要下载的时间范围
    DownLoad_Data=N_Data[index]                 #需要下载的数据量范围
    Down_TimeCopy=N[index]                       #未知用处
    Us = N_PlayData[index][0]  # 已经播放的数据量
    Ss = N_Data[index][0]      #在开始时刻已经下载的数据量
    Ut = {}  # 上限集合
    Lt = {}  # 下限集合

    start_time = N[index][0]
    end_time = N[index][1]
    ####在这里需要计算上限u=[].下限L=[]
    for j in range(start_time, end_time + 1):
        Ut[j] = min((j + 1 - start_time) * r + Us + w, N_Data[index][1])
        Lt[j] = max(Ss, (j + 1 - start_time) * r + Us)
        # print('time',j,Ut[j] ,Lt[j] )
    while v > 0:
        #############测试
        baseValue = 0
        baseStart = 0
        baseEnd = 0
        for k in range(start_time, end_time + 1):
            s_t, e_t, ni, s_data, e_data = bestRatio(k, end_time + 1, Ss, Ut, Lt)
            #  print(s_t,e_t,'能耗比为:',ni,'原有的最大能耗比为',baseValue)
            if ni > baseValue:
                baseValue = ni
                baseStart = s_t
                baseEnd = e_t
                start_data = s_data
                end_data = e_data
                print('   ')
            # print('现在最大的等效能耗值为', )

        print('计算结果，最大的等效能耗值为', baseValue, '对应的下载时间为', baseStart, baseEnd, '下载的数据量区间为', start_data, end_data)
        X_Time.append([baseStart, baseEnd])
        X_Data.append([start_data,end_data])
        print(X_Time)
        v=v-(end_data-start_data)

    ######改变需要下载的时间段,去掉已经下载的时间
        for i in Down_Time:
            print(Down_Time,DownLoad_Data)
            print(i)
            if i[0] <= baseStart and i[1] >= baseEnd:
                temp = i[:]
                Down_Time.remove(i)
                if i[0] != baseStart:
                    Down_Time.append([temp[0], baseStart - 1])
                if i[1] != baseEnd:
                    Down_Time.append([baseEnd + 1, temp[1]])


        index=index+1



        #改变上下限
        tem_Lt=Lt[:]
      #  for k in range()





'''
    for i in N_Data:
        index = 0
        v = i[1] - i[0]
        start_time = N[index][0]
        end_time = N[index][1]
        Us = N_PlayData[index][0]  # 已经播放的数据量
        Ss = N_Data[index][0]

        #############测试
        Ut = {}  # 上限集合
        Lt = {}  # 下限集合
        ####在这里需要计算上限u=[].下限L=[]
        for j in range(start_time, end_time + 1):
            Ut[j] = min((j + 1 - start_time) * r + Us + w, N_Data[index][1])
            Lt[j] = max(Ss, (j + 1 - start_time) * r + Us)
            # print('time',j,Ut[j] ,Lt[j] )
        baseValue = 0
        baseStart = 0
        baseEnd = 0
        for k in range(start_time, end_time + 1):
            s_t, e_t, ni, s_data, e_data = bestRatio(k, end_time + 1, Ss, Ut, Lt)
            #  print(s_t,e_t,'能耗比为:',ni,'原有的最大能耗比为',baseValue)
            if ni > baseValue:
                baseValue = ni
                baseStart = s_t
                baseEnd = e_t
                start_data = s_data
                end_data = e_data
                print('   ')
            # print('现在最大的等效能耗值为', )

        print('计算结果，最大的等效能耗值为', baseValue, '对应的下载时间为', baseStart, baseEnd, '下载的数据量区间为', start_data, end_data)
        X_Time.append([baseStart, baseEnd])
        print(X_Time)

'''




