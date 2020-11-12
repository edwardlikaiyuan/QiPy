# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:30:12 2019

@author: Administrator
"""
#引用部分
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

#数据清理
datafile = u'聚类test.xlsx'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
outfile = u'聚类_out.xlsx'#设置输出文件的位置
data = pd.read_excel(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
d = DataFrame(data)
d.head()


#聚类
mod = KMeans(n_clusters=4, init = 'k-means++',n_jobs = 5, max_iter = 50)#聚成3类数据,并发数为4，最大循环次数为500
mod.fit_predict(d)#y_pred表示聚类的结果

# 一些重要的参数：
# n_clusters = 8,          #聚类个数，K值，默认8
# init = 'k-means++',      #初始化类中心点选择方法，可选：
#         {
#             'k-means++', #是一种优化选择方法，比较容易收敛
#             'random',    #随机选择
#             an ndarray   #可以通过输入ndarray数组手动指定中心点
#         }
# max_iter:                #最大迭代数         
# precompute_distances：   #预计算距离，计算速度更快但占用更多内存。auto  True
# copy_x                   # True,原始数据不变，False直接在原始数据上做更改           



#聚成3类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis = 1)
r.columns = list(d.columns) + [u'类别数目']
print(r)

#给每一条数据标注上被分为哪一类
r = pd.concat([d, pd.Series(mod.labels_, index = d.index)], axis = 1)
r.columns = list(d.columns) + [u'聚类类别']
print(r.head())
r.to_excel(outfile)#如果需要保存到本地，就写上这一列



#可视化过程
from sklearn.manifold import TSNE
 
ts = TSNE()
ts.fit_transform(r)
ts = pd.DataFrame(ts.embedding_, index = r.index)
 
import matplotlib.pyplot as plt
 
a = ts[r[u'聚类类别'] == 0]
plt.plot(a[0], a[1], 'r.')
a = ts[r[u'聚类类别'] == 1]
plt.plot(a[0], a[1], 'go')
a = ts[r[u'聚类类别'] == 2]
plt.plot(a[0], a[1], 'b*')
a = ts[r[u'聚类类别'] == 3]
plt.plot(a[0], a[1], 'k+')
plt.show()