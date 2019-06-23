# -*- coding=utf8 -*-

from scrapy.cmdline import execute #可以执行scrapy脚本

import sys
import os
print (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))) #os.path.abspath(__file__)获取当前文件的路径 然后os.path.dirname是文件夹目录
execute(["scrapy","crawl","weibohuati"])

 