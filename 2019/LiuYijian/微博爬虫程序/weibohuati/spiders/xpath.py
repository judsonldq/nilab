from selenium import webdriver
import time
import requests
import re
import xlwt
from bs4 import BeautifulSoup

weibo=xlwt.Workbook(encoding='utf-8')
sheet=weibo.add_sheet('Sheet1')
def getcommentInfo(weibo_id_list, cookies, browser):

    # browser = webdriver.Chrome(executable_path="..\chromedriver.exe")
    # browser.add_cookie(cookies)
    person_id = []
    for weibo_id in weibo_id_list:
        url = "https://weibo.com/{0}".format(weibo_id)
        try:
            browser.set_page_load_timeout(300)
            browser.get(url)
            # time.sleep(1)
            # for i in range(0,10):
            #     browser.execute_script("window.scrollBy(0,5000)")
            #     time.sleep(1)

            # try:
            #     while 1:#点击加载更多
            #         browser.find_element_by_css_selector('#Pl_Official_WeiboDetail__60 > div > div > div > div.WB_feed_repeat.S_bg1.WB_feed_repeat_v3 > div > div.repeat_list > div:nth-child(2) > div > div > a > span').click()
            #         time.sleep(1)
            # except:
            #        pass

            time.sleep(2)
            r = browser.page_source
            soup = BeautifulSoup(r, "lxml")
            weibo_author_info = str(soup.find_all("div",class_="WB_info")[0])#bs4解析页面
            name = re.findall('>(.*?)</a>', weibo_author_info)[0]#昵称
            id = re.findall('id=(.*?)&', weibo_author_info)[0]#微博id
            person_id.append(id)
            print(name,id)
            weibotime = str(soup.find_all("div",class_="WB_from S_txt2")[0])
            weibocontent = str(soup.find_all("div", class_="WB_feed_detail clearfix")[0])
            weibobarinfo = str(soup.find_all("div", class_="WB_feed_handle")[0])
            weibo_text_model = '<div class="WB_text W_f14"[\s\S]*?</div>'
            weibo_text = re.findall(weibo_text_model, weibocontent)[0]
            weibo_text = deleteH5(weibo_text).strip()
            print(weibo_text)
            weibo_bar_model = "<li[\s\S]*?</li>"
            weibo_bar_list = re.findall(weibo_bar_model, weibobarinfo)
            zhuanfa = weibo_bar_list[1]
            pinglun = weibo_bar_list[2]
            zan = weibo_bar_list[3]
            timemodel = 'title="(.*?)"'
            wbtime = re.findall(timemodel,weibotime)[0]#获取时间
            try:
                zhuanfanum = re.findall('<em>(\d+)</em>', zhuanfa)[0]#获取转发数目
            except:
                zhuanfanum = 0
            try:
                pinglunnum = re.findall('<em>(\d+)</em>', pinglun)[0]#获取评论数目
            except:
                pinglunnum = 0
            try:
                zannum = re.findall('<em>(\d+)</em>', zan)[0]#获取点赞数目
            except:
                zannum = 0
            print(zhuanfanum, pinglunnum, zannum)
            import xlrd
            from xlutils.copy import copy
            book = xlrd.open_workbook(r"/Users/lyj/Desktop/数据2/服务.xlsx)#写入微博正文内容
            booknew = copy(book)
            sh = book.sheet_by_name('Sheet1')
            shnew = booknew.get_sheet(0)
            rows = sh.nrows
            shnew.write(rows, 0, name)
            shnew.write(rows, 1, id)
            shnew.write(rows, 2, weibo_text)
            shnew.write(rows, 3, zhuanfanum)
            shnew.write(rows, 4, pinglunnum)
            shnew.write(rows,5,zannum)
            shnew.write(rows,6,wbtime)
            booknew.save(r"/Users/lyj/Desktop/数据2/服务.xlsx")

            #写入评论信息表

        #     comment_list = soup.find_all("div", class_="list_li S_line1 clearfix")  # bs4解析页面
        #     # model = '<div class="repeat_list" node-type="feed_cate">[\s\S]*'
        #     # repeat = re.findall(model,r)[0]
        #     # comment_list = re.findall('<div comment_id=([\s\S]*?)list_ul', repeat)#提取所有评论信息
        #     print(len(comment_list))
        #     for comment in comment_list:
        #         comment = str(comment)
        #         name = re.findall('<img.*?alt="(.*?)"',comment)[0]#昵称
        #         id = re.findall('<img alt=".*?" src=".*?" usercard="id=(\d+)',comment)[0]#id
        #         comment_text = re.findall('<div class="WB_text">([\s\S]*?)</div>',comment)[0].strip()#评论内容
        #         try:
        #             like = re.findall('<em>(\d+)</em>',comment)[0]#点赞数
        #         except:
        #             like = 0
        #         comment_text = deleteH5(comment_text)#评论内容删h5元素
        #         comment_text = comment_text.replace(name+"：","")#评论内容
        #         timemodel1 = '<div class="WB_from S_txt2">(.*?)</div>'
        #         pltime = re.findall(timemodel1, comment)[0]#获得时间
        #         one = []
        #         one.append(name)
        #         one.append(id)
        #         one.append(comment_text)
        #         one.append(like)
        #         print(one)
        #         person_id.append(id)

        #         book = xlrd.open_workbook(r"/Users/lyj/Desktop/数据2/文化微博评论.xlsx")#写入微博评论信息
        #         booknew = copy(book)
        #         sh = book.sheet_by_name('Sheet1')
        #         shnew = booknew.get_sheet(0)
        #         rows = sh.nrows
        #         shnew.write(rows,0,name)
        #         shnew.write(rows,1,id)
        #         shnew.write(rows,2,comment_text)
        #         shnew.write(rows,3,like)
        #         shnew.write(rows,4,weibo_id)
        #         shnew.write(rows,5,pltime)
        #         booknew.save(r"/Users/lyj/Desktop/数据2/文化微博评论.xlsx")
        except:
            print("出现错误")
    # browser.close()
    return person_id
def deleteH5(text):
    model = '<.*?>'
    h5_list = re.findall(model,text)
    for h5 in h5_list:
        text = text.replace(h5,"")
    model = "&nbsp"
    list = re.findall(model,text)
    for i in range(0,len(list)):
        text=text.replace("&nbsp;","")

    model = "\r"
    list = re.findall(model, text)
    for i in range(0, len(list)):
        text=text.replace("\r", "")

    model = "\n"
    list = re.findall(model, text)
    for i in range(0, len(list)):
        text=text.replace("\n", "")
    return text
# getcommentInfo(url="https://weibo.com/3221447514/GlAQph8Ra",weiboid="1")


def getHtml(r,personid):
    information = []
    try:
        try:
            model = '昵称：.*<span class="pt_detail">([\s\S]*?)<'
            name = re.findall(model, r)[0]
        except:
            name = "未知"

        try:
            model = '所在地：.*<span class="pt_detail">([\s\S]*?)<'
            place = re.findall(model, r)[0]
            # print(place)
        except:
            place = "未知"
        try:
            model = '性别：.*<span class="pt_detail">([\s\S]*?)<'
            sex = re.findall(model, r)[0]
            # print(sex)
        except:
            sex = "未知"
        try:
            model = '注册时间[\s\S]*?<[\s\S]*?>[\s\S]*?<.*?>([\s\S]*?)</span>'
            registe_time = re.findall(model, r)[0].strip()
            # print(registe_time)
        except:
            registe_time = "未知"

        if "教育信息" in r:
            try:
                model ='教育信息[\s\S]*?ul>'
                text = re.findall(model,r)
                text = text[0]
                model_school = ">(.*?)\\/a"
                school_list = re.findall(model_school,text)
                for i in range(0,len(school_list)):
                    school_list[i] = school_list[i][:-3]
                # print(";".join(school_list))
                school_str = ";".join(school_list)

            except:
                school_str = "未知"
        else:
            school_str = "未知"
        if "生日" in r:
            try:
                model='生日[\s\S]*?<span class="pt_detail">([\s\S]*?)<'
                birth = re.findall(model,r)[0]
            except:
                birth = "未知"
        else:
            birth="未知"
        import xlrd
        from xlutils.copy import copy
        book = xlrd.open_workbook("作者信息.xlsx")
        booknew = copy(book)
        sh = book.sheet_by_name('Sheet1')
        shnew = booknew.get_sheet(0)
        rows = sh.nrows
        shnew.write(rows, 0, name)
        shnew.write(rows, 1, personid)
        shnew.write(rows, 2, place)
        shnew.write(rows, 3, sex)
        shnew.write(rows, 4, registe_time)
        shnew.write(rows, 5, school_str)
        shnew.write(rows,6,birth)
        booknew.save("作者信息.xlsx")
        print("成功插入作者信息")


    except:
        print("作者信息插入异常")