from selenium import webdriver
import time
import threading
import pandas as pd
import os 

# 请求dash.js开源网页客户端
browser = webdriver.Firefox(executable_path= '/home/zdf/桌面/20181231/dashVideoSpider/geckodriver')
#url = 'http://reference.dashif.org/dash.js/nightly/samples/dash-if-reference-player/index.html'
url = 'http://reference.dashif.org/dash.js/v2.9.3/samples/dash-if-reference-player/index.html'
browser.get(url)
# 待网页加载完成，等待3s
time.sleep(3)
# 加载目标视频链接
input = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/input')
input.clear()

# 兔子、松鼠、蝴蝶 （10种码率 10min）
#input.send_keys('http://dash.akamaized.net/akamai/bbb_30fps/bbb_30fps.mpd')

# 运动集锦 （2种码率 2500-4000 6min） 
#input.send_keys('http://dash.akamaized.net/dash264/TestCases/5a/1/manifest.mpd')

# 大象的梦 （4种码率 150-750 10min）
# input.send_keys('http://dash.akamaized.net/dash264/TestCases/1a/netflix/exMPD_BIP_TC1.mpd')
# 大象的梦 （3种码率 1621-4076 10min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCases/4b/qualcomm/1/ED_OnDemand_5SecSeg_Subtitles.mpd')
# 大象的梦 （3种码率 1197-4102 10min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCases/2a/qualcomm/1/MultiResMPEG2.mpd')
# 大象的梦 （4种码率 1197-7952 10min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCasesHD/2a/qualcomm/1/MultiResMPEG2.mpd')


#未来世界 （5种码率 480-3352 12min）
#input.send_keys('http://demo.unified-streaming.com/video/tears-of-steel/tears-of-steel-tiled-thumbnails.mpd')
#未来世界 （386-1117 12min）
#input.send_keys('http://media.axprod.net/TestVectors/v7-MultiDRM-MultiKey/Manifest_1080p.mpd')
#未来世界 （5种码率 386-2773 12min）
#input.send_keys('http://media.axprod.net/TestVectors/v7-Clear/Manifest_1080p.mpd')
#未来世界 （391-5425 12min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCasesIOP33/adapatationSetSwitching/5/manifest.mpd')
#未来世界 （3种码率 751-4006 12min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCases/4b/qualcomm/2/TearsOfSteel_onDem5secSegSubTitles.mpd')

#纪录片 （2种码率 3000-4000 5min）
#input.send_keys('http://dash.akamaized.net/dash264/TestCasesHD/2b/DTV/1/live.mpd')

#电视屏保 （1种码率 1144 5min）
input.send_keys('http://dash.akamaized.net/dash264/CTA/imsc1/IT1-20171027_dash.mpd')

# 动态画面 花花草草（6种码率 2859-19683 3min）
#input.send_keys('http://dash.akamaized.net/akamai/streamroot/050714/Spring_4Ktest.mpd')

# time.sleep(2)
# 点击“show Options”按钮
show_optins_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[1]')
show_optins_button.click()
time.sleep(2)
# 选择ABR策略
# 选择“ABR Stategy：BOLA”
bola_button = browser.find_element_by_xpath('//*[@id="abrBola"]')
bola_button.click()
time.sleep(1)
# 点击“Hide Options”按钮
hide_optins_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[1]')
hide_optins_button.click()
time.sleep(1)
# 打开wireshark
os.system("gnome-terminal -e 'wireshark'")
time.sleep(5)

# 自动点击load按钮，开始视频请求（保存视频请求时刻）
load_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[3]')
load_button.click()
requestTime = time.time()
# 开始抓取数据，每一个抓取周期的时间间隔是1s
# 每周期抓取到的数据组成一个列表，存到一个大列表里
videoClientData = []

# spider功能函数：用于定位页面中制定元素，周期性摘取该页面元素的数据。
def spider():
    videoStates = []
    
    greenwitchTime = time.time()
    #input = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[1]/text()')
    bufferLength = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[1]').text
    bitrateDownloading = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[2]').text
    indexDownloading = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[3]').text
    currentIndex_maxIndex = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[4]').text
    droppedFrames = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[5]').text
    latencyMinAvgMax = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[6]').text
    downloadMinAvgMax = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[7]').text
    ratioMinAvgMax = browser.find_element_by_xpath('//*[@id="videoStatsTab"]/div/div[8]').text

    videoStates = [greenwitchTime, bufferLength, bitrateDownloading, indexDownloading, currentIndex_maxIndex, droppedFrames, latencyMinAvgMax, downloadMinAvgMax, ratioMinAvgMax]
    videoClientData.append(videoStates)
    global timer
    timer = threading.Timer(0.25, spider)
    timer.start()

    return videoClientData

# 设置定时器采集时间，到了时间关闭定时器。数据采集结束
timer = threading.Timer(0.25, spider)
timer.start() 
'''
# 正常演播3min
time.sleep(180)
# 新打开一个终端，输入命令"cbm",打开网络带宽监控软件
os.system("gnome-terminal -e 'bash -c \"cbm; exec bash\"'")
# 新打开一个终端，输入命令限制带宽指令：sudo wondershaper eno1 20 50
# 将网卡eno1的下行带宽和上行带宽分别改成：20kbps、50kbps
os.system("gnome-terminal -e 'bash -c \"sudo wondershaper eno1 0 1000; exec bash\"'")
time.sleep(90)
os.system("gnome-terminal -e 'bash -c \"sudo wondershaper -c eno1; exec bash\"'")
'''
time.sleep(280)
timer.cancel() 

#将采集到的视频数据：list >>> dataframe >>> csv
name = ['greenwitchTime','bufferLength', 'bitrateDownloading',
        'indexDownloading', 'currentIndex_maxIndex', 
       'droppedFrames', 'latencyMinAvgMax', 'downloadMinAvgMax',
       'ratioMinAvgMax']
test = pd.DataFrame(columns = name, data = videoClientData)
test.to_csv('/home/zdf/桌面/20181231/data/text.csv')