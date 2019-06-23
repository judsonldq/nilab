import numpy
import pandas

file=open("C:\\Users\wangwei\Desktop\data1.txt")
data=file.readlines()
band=[ ]
for i in data:
    row=i.split('\n')
    row=float(row[0])
    band.append(row)

print('最大带宽',max(band),'最小带宽',min(band),'平均带宽',sum(band)/len(band),len(band))
T=1800
bandWish=band[:]
r=3     #播放速率为1M/s
w=r*60*2
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
Cut_Data={}           #从发生卡顿的第一秒开始记录每一秒下载的数据量，直到卡顿结束
Cut_PlayData={}

while down_data<=VideoData:                                #有可能下载完毕但是播放没有完毕
    if down_data==0 or down_data+bandWish[t]<playData+r:   #发生卡顿,只下载不播放
        rebuTime[t]="C"
        Cut_Data[t]=down_data
        Cut_PlayData[t]=playData
        Data.append(down_data)
        Play_Data.append(playData)
        re_cut[t]="C"
        T_W=down_data-playData       #当前时刻缓存区内的数据
      #  print('发生卡顿', '卡顿时间:', t, ' ', '缓存区数据量', T_W,'已经下载的数据',down_data,'已经播放的',playData)
        while T_W+bandWish[t]<w:
            T_W=T_W+bandWish[t]
            down_data=down_data+bandWish[t]
            Cut_Data[t]=down_data
            Cut_PlayData[t]=playData
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

print('缓存区发生卡断的时间',re_cut)
print('缓存区发生上溢和下溢时间',rebuTime)
print('对应上时间已经下载完成的数据量',Data)
print('对应上时间已经播放的数据量',Play_Data)

print('发生卡顿时期已经下载的数据量',Cut_Data)
print('发生卡顿时期已经播放的数据量',Cut_PlayData)
#print('长度为',len(Cut_Data))







N=[[64,1647]]                       #可优化段
N_Data=[[360,4839]]                 #可优化段时刻对应的已经下载完成的数据量
N_PlayData=[[0,4479]]                                  #可优化段对应的已经播放的数据量

##############开始测试：
#N=[[10,1745]]
#N_Data=[[ 58.3625,1800]]
#N_PlayData=[[0,1742]]


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
   # print('   ')
  #  print('开始时间',start_t,'结束时间',end_t,'开始时已经下载的数据量',start_d,'开始时刻上限',u[start_t],'开始时刻下限',v[start_t])
    if u[start_t]<=v[start_t]:
        print('这里错误',)
        print('开始时间', start_t, '结束时间', end_t, '开始时已经下载的数据量', start_d, '开始时刻上限', u[start_t], '开始时刻下限', v[start_t])
        exit()
    bsData=max(start_d,v[start_t])                                 #开始时刻的数据量 max(开始时刻缓存下限，开始时刻已经下载的数据量)
    beData=bsData                #开始时刻的数据量
   # print('开始时刻下载的数据量', bsData,'结束时刻初始化下载的数据量',beData)
    for i in range(start_t,end_t):
        bsData=bsData+bandWish[i]                  #一会判断一下时start_t还是start_t-1
     #   print (i,'时刻的信道',bandWish[i],'下载后的数据量',bsData,"上限",u[i],'下限',v[i])
        if bsData>=u[i] or bsData>=VideoData:
            ActallyEnd_time=i
            bsData=u[i]
            break
        elif bsData<v[i] or i==end_t-1:
            #break
         #   print('发生卡顿,下载停止')
            return -1,-1,-1,-1,-1,-1,-1
    ####计算KI
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

    if ki<=0:
        print("出现错误")
        print(start_t,ActallyEnd_time)
        print(c)
        exit()

    else:
       # print('开始时的数据量:',beData,'结束时的数据量',bsData,'消耗的总时间',ki)
        ni=(bsData-beData)/ki
        extra_time=ki-(ActallyEnd_time-start_t)
      #  print('原来时间',ActallyEnd_time-start_t,'现在时间',ki,'多余时间',ki-(ActallyEnd_time-start_t))
    #print('实际开始时间',start_t,'实际结束时间',ActallyEnd_time,'开始时已经下载的数据量:',beData,'实际结束时的下载的数据量',bsData,)
    #################这里的实际结束时间，在这一秒种并没有下载
   # print('能耗比',ni)
    if bsData<beData:
        print('错误',bsData,beData)
        exit()
   # print(ni)
    if ni<0:
        print('出现错误')
        exit()
    return start_t,ActallyEnd_time,ni,beData,bsData,extra_time,ki                             #####测试,返回下载的数据量段范围



def main_H():
    Video_Time=0
    Tail_Count = 0
    index = 0

    for i in N_Data:
        v = N_Data[index][1] - N_Data[index][0]
        Down_Time = [N[index]]  # 需要下载的时间范围
        DownLoad_Data = [N_Data[index]]  # 需要下载的数据量范围
        Down_TimeCopy = N[index]  # 未知用处
        Us = N_PlayData[index][0]  # 已经播放的数据量
        Ut = {}  # 上限集合
        Lt = {}  # 下限集合
        start_time = N[index][0]
        end_time = N[index][1]
        ####在这里需要计算上限u=[].下限L=[]
        Ss = N_Data[index][0]  # 在开始时刻已经下载的数据量
        Se = N_Data[index][0]  # 在结束时刻需要下载的数据量
        for j in range(start_time, end_time + 1):
            Ut[j] = min((j + 1 - start_time) * r + Us + w, N_Data[index][1])
            Lt[j] = max(Ss, (j + 1 - start_time) * r + Us)
            # print('time',j,Ut[j] ,Lt[j] )
        all_data = 0
        count = 0
        while v > 0:
            start_time = Down_Time[0][0]
            end_time = Down_Time[0][1]
            Ss = DownLoad_Data[0][0]
            #############测试
            baseValue = 0
            baseStart = 0
            baseEnd = 0
          #  print('开始下载',start_time,end_time)
            for k in range(start_time, end_time + 1):
                s_t, e_t, ni, s_data, e_data,ex_t,a_Time= bestRatio(k, end_time + 1, Ss, Ut, Lt)

                if ni > baseValue:
                   # print('TRUE!!!!!!是现有的最大能耗比！！！！！',s_t ,e_t,s_data,e_data,ni)
                    baseValue = ni
                    baseStart = s_t
                    baseEnd = e_t
                    start_data = s_data
                    end_data = e_data
                    extra_t=ex_t
                    all_time=a_Time
                    # print('   ')
                # print('现在最大的等效能耗值为', )
           # print('  ')
            #print('计算结果，最大的等效能耗值为', baseValue, '对应的下载时间为', baseStart, baseEnd, '下载的数据量区间为', start_data, end_data)
           # print('开始时刻下限',Lt[baseStart],'开始时刻已经下载的数据量',start_data )
            #print('结束时刻上限',Ut[baseEnd],'结束时刻已经下载的数据量',end_data)

          #  print('额外的时间',extra_t,'总时间',all_time)
            X_Time.append([baseStart, baseEnd])
            X_Data.append([start_data, end_data])
            v = v - (end_data - start_data)
            all_data = all_data + (end_data - start_data)
            Tail_Count=Tail_Count+extra_t
            Video_Time=Video_Time+all_time
          #  print('已经下载完成的时间为',X_Time)
            ######改变需要下载的时间对应的数据量范围
            OverPre = -1
            OverNext = -1
            for i in DownLoad_Data:
                if i[0] <= start_data and i[1] >= end_data:
                    temp2 = i[:]
                    DownLoad_Data.remove(i)
                    if i[0] != start_data:
                        DownLoad_Data.append([temp2[0], min(start_data, temp2[1])])
                    elif i[0] == start_data:
                        OverPre = 0
                    if i[1] != end_data:
                        DownLoad_Data.append([end_data, temp2[1]])
                    elif i[1] == end_data:
                        OverNext = 0
            ######改变需要下载的时间段,去掉已经下载的时间
            for i in Down_Time:
                if i[0] <= baseStart and i[1] >= baseEnd:
                    temp = i[:]
                    Down_Time.remove(i)
                    update_u = []
                    update_l = []
                    if i[0] != baseStart and OverPre != 0:
                        Down_Time.append([temp[0], baseStart - 1])
                        update_u = [temp[0], baseStart - 1]
                    if i[1] != baseEnd and OverNext != 0:
                        Down_Time.append([baseEnd + 1, temp[1]])
                        update_l = [baseEnd + 1, temp[1]]

            Down_Time = sorted(Down_Time)
            DownLoad_Data = sorted(DownLoad_Data)
            #print('需要下载的时间范围为', Down_Time, '对应的数据量范围为', DownLoad_Data)
         #   print('仍需下载的数据时间',Down_Time,'仍需下载的数据量',DownLoad_Data)
            if len(Down_Time) != len(DownLoad_Data):
                print('长度出现错误',len(Down_Time),len(DownLoad_Data))
                print(Down_Time)
              #  print(' ')
             #   print(DownLoad_Data)
              #  print('一共需要下载的数据量')
                break
            #  print('一共需要下载的数据量为',1800-60,'仍需下载的数据量为',v,'已经下载的时间为',b)
          #  print('已经下载完成的时间',X_Time,'对应的数据量范围',X_Data)


            ##########改变上下限
            if len(update_u) > 0:
                for i in range(update_u[0] + 1, update_u[1]):
                    Ut[i] = min(Ut[i], start_data)
            if len(update_l) > 0:
                for i in range(update_l[0] + 1, update_l[1]):
                    Lt[i] = max(end_data, Lt[i])

            # 判断上限是否超出               #######测试
            for i in range(0, len(DownLoad_Data)):
                if DownLoad_Data[i][0] > Ut[Down_Time[i][0]]:
                    print('上限超出错误,时间：', Down_Time[i][0], '已经下载的', DownLoad_Data[i][0], '上限', Ut[Down_Time[i][0]])
                    break
                    exit()
            count=count+1
            #if count==1:
            #    break
            #    exit()
          #  print('   ')
        return X_Time,X_Data,Tail_Count,Video_Time



X_Time,X_Data,Tail_Count,V_T=main_H()
c = 0
for i in X_Data:
    c = c + i[1] - i[0]
b = 0
for time in X_Time:
    b = b + time[1] - time[0]

print('  ')
print('EDA下载传输完毕','下载总时间为',V_T+N[0][0],'下载的总数据量',c+N_Data[0][0])
print('额外消耗的Tail_time时间为',Tail_Count)


# print(all_data)
#print(X_Data)
#print('  ')
#print(X_Time)


