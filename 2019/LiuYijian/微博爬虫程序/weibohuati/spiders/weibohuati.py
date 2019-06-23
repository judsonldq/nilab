# -*- coding=utf8 -*-

import re
import json
import datetime
import random
import pickle
import time
from urllib import parse
from bs4 import BeautifulSoup
from weibohuati.spiders.xpath import getcommentInfo,getHtml



import requests
import scrapy   
from scrapy.loader import ItemLoader


class weibohuatispider(scrapy.Spider):
    name = "weibohuati"
    allowed_domains = ["www.weibo.com"]
    start_urls = "https://s.weibo.com/weibo/%25E5%258C%2597%25E4%25BA%25AC%2520%25E5%25A4%2584%25E7%2590%2586?q=北京%20污泥&typeall=1&suball=1&timescope=custom:2018-10-01:2018-12-31&Refer=g&page={0}"
    #start_urls = "https://s.weibo.com/weibo?q=%E6%96%87%E5%8C%96%E9%81%97%E4%BA%A7%E4%BF%9D%E6%8A%A4&typeall=1&suball=1&timescope=custom:2018-01-01:2017-01-31&Refer=SWeibo_box&page={0}"
    headers = { #访问头
        "HOST": "www.weibo.cn",
        "Referer": "http://www.weibo.cn",
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64; rv:59.0) Gecko/20100101 Firefox/59.0"
    }

    custom_settings = {
        "COOKIES_ENABLED": True
    }
    cookie_dict = {}
    index = 1 #url参数
    def parse(self, response):
        """
        提取出html页面中的所有url 并跟踪这些url进行下一步的爬
        """
        comment_list = []
        comment_list = response.css("div[class='c']::text").extract()
        while ':转发微博' in comment_list:
            comment_list.remove(':转发微博')


        print ("第{0}个页面已经爬取完成".format(self.index))
        self.index += 1
        import time
        time.sleep(0.1)
        yield scrapy.Request(self.start_urls.format(self.index),callback=self.parse,dont_filter=True)
    def start_requests(self):
        from selenium import webdriver
        from scrapy.selector import Selector
        from scrapy import Request
        import time

        browser = webdriver.Chrome()
        browser.set_page_load_timeout(300)
        browser.get("https://weibo.com/")  # 浏览器打开页面
        time.sleep(5)
        browser.maximize_window()
        browser.find_element_by_css_selector("#loginname").send_keys("18435132420")
        browser.find_element_by_css_selector("#pl_login_form > div > div:nth-child(3) > div.info_list.password > div > input").send_keys("lyjglyjn25")
        time.sleep(10)
        browser.find_element_by_css_selector("#pl_login_form > div > div:nth-child(3) > div.info_list.login_btn > a").click()  # 模拟点击登陆按钮
        time.sleep(4)  # 等待加载

        Cookies = browser.get_cookies()
        print(Cookies)

        for cookie in Cookies:#保存cookie
            f = open('./' + cookie['name'] + '.weibo', 'wb')
            pickle.dump(cookie, f)
            f.close()
            self.cookie_dict[cookie['name']] = cookie['value']

        while self.index<=49:#爬取50页
            url = self.start_urls.format(self.index)
            print(url)
            browser.get(self.start_urls.format(self.index))
            print(browser.page_source)
            self.index += 1
            browser.execute_script("window.scrollBy(0,3000)")#模拟下拉
            # time.sleep(1)
            browser.execute_script("window.scrollBy(0,5000)")
            # time.sleep(1)
            browser.execute_script("window.scrollBy(0,3000)")
            # time.sleep(1)
            # browser.execute_script("window.scrollBy(0,5000)")
            # time.sleep(1)
            # browser.execute_script("window.scrollBy(0,5000)")
            # time.sleep(1)

            text = browser.page_source
            soup = BeautifulSoup(text, "lxml")
            info_list = soup.find_all("div", class_="card-wrap")
            import xlrd
            from xlutils.copy import copy
            import re
            weibo_id_list = []
            for info in info_list:
                try:
                    info =str(info)
                    weibo_id_model = 'href="//weibo.com/(.*?)?refer_flag='
                    weibo_id = re.findall(weibo_id_model, info)[2][:-1]
                    weibo_id_list.append(weibo_id)#获取微博id
                except:
                    print("出异常了")
            # for weibo in weibo_id_list:
            #     print("页数"+str(self.index))
            personlist = getcommentInfo(weibo_id_list, cookies=self.cookie_dict, browser=browser)#获取具体页面信息
            # for person in personlist:
            #     browser.get("https://weibo.com/p/100505{0}/info?mod=pedit_more".format(person))
            #     time.sleep(1.5)
            #     getHtml(browser.page_source,person)
            #     time.sleep(1)








        # return [scrapy.Request(url=self.start_urls.format(self.index), dont_filter=True, cookies=self.cookie_dict)]

