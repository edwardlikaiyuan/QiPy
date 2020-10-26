# encoding =utf-8
"""
Spyder Editor

OLDLee.
2019年12月16日修改
"""
#引用部分
import json
# from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import sys
from scipy import interpolate

# a=sys.argv[1]
# a='2019121614'

#读取配置文件
# with open('setDir.ini','r')as fset:
#     path=fset.readline().strip('\n').replace('\\','/')
#读取json数据
with open('../data/DM.json', 'r',encoding='utf-8') as fdm:
    ddm = json.load(fdm)
    dfdm = pd.json_normalize(ddm['DS'])
dfdm=dfdm.astype('float64')
dfdm['Station_Id_d']=dfdm['Station_Id_d'].astype(np.int)    #把站号弄成整型数据
df=dfdm.replace([999999,999999.0,999998,999998.0,999990,999990.0],[99999,99999,99999,99999,0.01,0.01])
# ……
# ……