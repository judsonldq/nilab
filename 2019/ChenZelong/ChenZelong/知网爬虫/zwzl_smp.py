import time
import requests
import json
import re
import csv
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from lxml import etree
from urllib.parse import urlparse, urljoin

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Host": "kns.cnki.net",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
}


class WebSetState:
    def __init__(self, url, headers, parent):
        self.url = url
        self.cookies = {}
        self.headers = headers
        self.parent = parent
        furl = urlparse(url)
        self.base_url = furl.scheme+"://"+furl.netloc
        self.init_cookie()

    def init_cookie(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(chrome_options=chrome_options)
        driver.get(self.url)
        time.sleep(5)

        driver.find_element_by_id('txt_1_value1').value = '电子'
        driver.find_element_by_id('btnSearch').submit()
        time.sleep(5)
        self.cookies = {item['name']: item['value']
                        for item in driver.get_cookies()}

    def get(self, url, **kwargs):
        url = urljoin(self.base_url, url)
        p = {}
        if 'params' in kwargs.keys():
            p = kwargs['params']
        headers = self.headers
        if 'headers' in kwargs.keys():
            headers = kwargs['headers']

        while True:
            try:
                r = requests.get(url, headers=headers,
                                cookies=self.cookies, params=p)
                break
            except:
                print('断线重连!')
                self.init_cookie()

        if r.status_code == 200:
            vf = requests.utils.dict_from_cookiejar(r.cookies)
            self.cookies = {**self.cookies, **vf}

        return r

    def post(self, url, data, **kwargs):
        url = urljoin(self.base_url, url)
        headers = self.headers
        if 'headers' in kwargs.keys():
            headers = kwargs['headers']
        
        while True:
            try:
                r = requests.post(url, headers=headers,
                            cookies=self.cookies, data=data)
                break
            except:
                print('断线重连!')
                self.init_cookie()
            
        if r.status_code == 200:
            vf = requests.utils.dict_from_cookiejar(r.cookies)
            self.cookies = {**self.cookies, **vf}
        return r


class Zwzl:
    def __init__(self, headers, keyword, filename):
        self.keyword = keyword
        self.website = WebSetState(
            "http://kns.cnki.net/kns/brief/result.aspx?dbprefix=SCOD#", headers, self)
        self.update_cookie()
        self.csv_writer = CsvWriter(filename)
        self.list_params = {
            "curpage": 0,
            "RecordsPerPage": "20",
            "QueryID": "20",
            "ID": "",
            "turnpage": "1",
            "tpagemode": "L",
            "dbPrefix": "SCOD",
            "Fields": "",
            "DisplayMode": "listmode",
            "PageName": "ASP.brief_result_aspx",
            "Param": "专利类别代码='1'",
            "isinEn": "0",
        }

        self.headers = {
            "Host": "kns.cnki.net",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "",
        }
        self.zl_headers = {
            "Accept":"*/*",
            "Accept-Encoding":"gzip, deflate",
            "Accept-Language":"zh-CN,zh;q=0.9",
            "Content-Length":"666",
            "Content-Type":"application/x-www-form-urlencoded",
            "Host":"dbpub.cnki.net",
            "Origin":"http://dbpub.cnki.net",
            "Proxy-Connection":"keep-alive",
            "Referer": "",
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
            "X-Requested-With":"XMLHttpRequest",
        }
        self.count = 0

    def update_cookie(self):
        self.website.init_cookie()
        data = {
                "action": "",
                "NaviCode": "*",
                "ua": "1.21",
                "isinEn": "0",
                "PageName": "ASP.brief_result_aspx",
                "DbPrefix": "SCOD",
                "DbCatalog": "专利数据总库",
                "ConfigFile": "SCOD.xml",
                "db_opt": "SCOD",
                "db_value": "中国专利数据库,国外专利数据库",
                "txt_1_sel": "SU$%=|",
                "txt_1_value1": self.keyword,
                "txt_1_relation": "#CNKI_AND",
                "txt_1_special1": "=",
                "his": "0",
            }
        self.website.post('http://kns.cnki.net/kns/request/SearchHandler.ashx', data = data)
        self.website.post('http://kns.cnki.net/kns/request/GetWebGroupHandler.ashx', data = data)

    def run(self):
        while True:
            self.list_params['curpage'] += 1
            print('爬取第{0}页'.format(self.list_params['curpage']))
            r = self.website.get(
                'http://kns.cnki.net/kns/brief/brief.aspx', params=self.list_params)
            self.getList(r)

    def getList(self, r):
        headers = self.headers.copy()
        headers['Referer'] = r.url
        tree = etree.HTML(r.content.decode())
        a_hrefs = tree.xpath('//a[@class="fz14"]/@href')
        for href in a_hrefs:
            detail_page = self.website.get(href, headers = headers)
            self.getDetail(detail_page)
        if len(a_hrefs) == 0:
            self.update_cookie()
            time.sleep(10)

    def getDetail(self, r):
        print("正在爬取:{0}".format(r.url))
        html = r.content.decode()

        self.getDataToCsv(r)
        
    def getDataToCsv(self, r):
        result = {}
        html = r.content.decode()
        result['url'] = r.url
        result['专利名称'] = find('<title>(.*?)--.*?</title>', html)
        result['申请人'] = find('【申请人】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)
        result['发明人'] = find('【发明人】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)

        result['摘要'] = find('【摘要】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)
        result['摘要'] = result['摘要'].replace(',','').replace('\n','').replace('\r', '')

        result['主权项'] = find('【主权项】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)
        result['主权项'] = result['主权项'].replace(',','').replace('\n','').replace('\r', '')
        
        result['主分类号'] = find('【主分类号】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)
        result['专利分类号'] = find('【专利分类号】.*?</td>[\s\S]*?<td.*?>.*?&nbsp;?(.*?)</td>', html)

        zl1_data = self.getVaildJson('zzfx_data = (\{.*?\})', html)
        result['相关专利'] = self.getOthersZLDataLList(r.url, 'http://dbpub.cnki.net/grid2008/dbpub/Detail.aspx?action=node&dbname=scpd&block=SCPD_ZZFX', zl1_data)
        zl2_data = self.getVaildJson('cgbz_data = (\{.*?\})', html)
        result['科技成果'] = self.getOthersZLDataLList(r.url, 'http://dbpub.cnki.net/grid2008/dbpub/Detail.aspx?action=node&dbname=scpd&block=SCPD_CGBZ', zl2_data)
        self.csv_writer.writerow(result)
        
    def getVaildJson(self, pattern, html):
        js = find(pattern, html)
        if js == 'none':
            return {'error':'0'}
        try:
            data = json.loads(zl_json_handle(js))
            return data
        except:
            return {}
        

    def getOthersZLDataLList(self, referer, url, data):
        self.zl_headers['Referer'] = referer
        r = self.website.post(url, data, headers = self.zl_headers)
        tree = etree.HTML(r.content.decode())
        if tree is None:
            return
        a_names = tree.xpath('//a[contains(@href,"detail.aspx?")]/text()')
        return ' '.join(a_names)

    def getOthersZL(self, referer, url, data):
        self.zl_headers['Referer'] = referer
        r = self.website.post(url, data, headers = self.zl_headers)

        tree = etree.HTML(r.content.decode())
        if tree is None:
            return
        a_hrefs = tree.xpath('//a[contains(@href,"detail.aspx?")]/@href')
        a_hrefs = list([urljoin('http://dbpub.cnki.net/grid2008/dbpub/', href) for href in a_hrefs])
        for href in a_hrefs:
            detail_page = self.website.get(href, headers = self.headers)
            print('爬取内部专利链接: ' + detail_page.url)
            self.getDataToCsv(detail_page)

class CsvWriter:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8', newline='')
        self.csv_writer = csv.writer(self.file)  
        self.is_header = False

    def writerow(self, data):
        if self.is_header == False:
            self.csv_writer.writerow(list(data.keys()))
            self.is_header = True
        self.csv_writer.writerow(list(data.values()))

    def __del__(self):
        self.csv_writer.close()
        self.file.close()

def find(pattern, string):
    result = re.findall(pattern, string, re.DOTALL)
    if len(result) > 0:
        return result[0]
    return 'none'

def zl_json_handle(js):
    s = js.split(',')
    for i in range(0, len(s)):
        if s[i].find("'+'") == -1:
            s[i] = s[i].replace("'", '"')
    return ','.join(s)

if __name__ == '__main__':
    Zwzl(headers, "电子", '电子.csv').run()
