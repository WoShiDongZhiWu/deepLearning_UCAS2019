'''
author,武栋，20190527
功能，将0-4499写到文本中，每个数字一行
'''

import os

with open('./Dataset/annotations/trainval.txt','a') as f:
    for i in range(4500):
        f.write('%a'%(i)+'\n')
    f.close()