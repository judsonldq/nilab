#file=open("C:\\Users\wangwei\Desktop\毕业设计数据\playback_progress_car_A_2017.12.11_15.04.57_R2000_W300.txt")
file=open("C:\\Users\wangwei\Desktop\毕业设计数据\two.txt")

data=file.read().split( )
ylable=[]
for i in data:
    ylable.append(float(i)/128)
print('最小带宽',min(ylable))    #最小带宽
print('最大带宽',max(ylable))
print('平均带宽',(sum(ylable)/len(ylable)))   #平均带宽
xlable=[i for i in range(0,len(ylable))]
print(len(ylable),'len')
with open("C:\\Users\wangwei\Desktop\data.txt",'w') as f2:
    for i in ylable:
        f2.writelines(str(i))
        f2.writelines('\n')