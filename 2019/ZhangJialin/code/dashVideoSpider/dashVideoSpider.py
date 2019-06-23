from selenium import webdriver
import time
import threading
import pandas as pd

#请求dash.js开源网页客户端
#browser = webdriver.Chrome(executable_path= '/Users/dave/anaconda3/chromeDriver/chromedriver')
#browser = webdriver.Firefox(executable_path= '/Users/dave/anaconda3/firefoxDriver/geckodriver')
browser = webdriver.Firefox(executable_path= '/home/zdf/桌面/20181231/dashVideoSpider/geckodriver')
url = 'http://reference.dashif.org/dash.js/nightly/samples/dash-if-reference-player/index.html'
browser.get(url)

#待网页加载完成，等待3s
time.sleep(3)
#加载目标视频链接
input = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/input')
input.clear()
#加密请求https
#input.send_keys('https://dash.akamaized.net/akamai/bbb_30fps/bbb_30fps.mpd')
#未加密请求http
input.send_keys('http://dash.akamaized.net/akamai/bbb_30fps/bbb_30fps.mpd')
#time.sleep(2)
#点击“show Options”按钮
show_optins_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[1]')
show_optins_button.click()
time.sleep(2)
#选择ABR策略
'''
#Dynamic_button = browser.find_element_by_xpath('//*[@id="abrDynamic"]')
#Dynamic_button.click()
#time.sleep(2)
'''
#选择“ABR Stategy：BOLA”
bola_button = browser.find_element_by_xpath('//*[@id="abrBola"]')
bola_button.click()
time.sleep(2)
'''
#选择“ABR Stategy：Throughput”
#throughput_button = browser.find_element_by_xpath('//*[@id="abrThroughput"]')
#throughput_button.click()
#time.sleep(2)
#点击“Hide Options”按钮
'''
hide_optins_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[1]')
hide_optins_button.click()
time.sleep(1)

#自动点击load按钮，开始视频请求（保存视频请求时刻）
load_button = browser.find_element_by_xpath('/html/body/div[2]/div[2]/div/span/button[3]')
load_button.click()
requestTime = time.time()
print (requestTime)

#开始抓取数据，每一个抓取周期的时间间隔是1s
#每周期抓取到的数据组成一个列表，存到一个大列表里
videoClientData = []
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


#设置定时器采集时间，到了时间关闭定时器。数据采集结束
timer = threading.Timer(0.25, spider)
timer.start() 
time.sleep(600)
timer.cancel() 
finishTime = time.time()
print (finishTime)

#将采集到的视频数据：list >>> dataframe >>> csv
name = ['greenwitchTime','bufferLength', 'bitrateDownloading',
        'indexDownloading', 'currentIndex_maxIndex', 
       'droppedFrames', 'latencyMinAvgMax', 'downloadMinAvgMax',
       'ratioMinAvgMax']
test = pd.DataFrame(columns = name, data = videoClientData)
test.to_csv('/Users/dave/Desktop/DASH-video_workshop/dataer/text.csv')