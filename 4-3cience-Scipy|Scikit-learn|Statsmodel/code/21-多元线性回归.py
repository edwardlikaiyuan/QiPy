# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:48:51 2019
基于DEM的气温插值方法
@author: Administrator
"""
#引用部分
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json

from sklearn.model_selection import train_test_split #引用样本分集交叉验证
from sklearn.linear_model import LinearRegression  #线性回归

import imageio
import maskout

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#显示中文&负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#DEM高程信息读入
#filename='./srtm_58_06_Clip1_1.tif'
tif16 = imageio.imread('../shp/srtm_58_06_Clip1_1.tif')

# a=sys.argv[1]
a='20190624000000'

#读取数据→数据清洗
#读取全部数据，删除异常数据→data   用于制作MLR模型
#按照行政区代码挑选出达州市内部站点信息DZdf，用于最后的站点填值
with open('../data/TemperRH.json', 'r',encoding='utf-8') as fTR:
    dd = json.load(fTR)
    df = json_normalize(dd['DS'])
#df = df.convert_objects(convert_numeric=True)
df['Admin_Code_CHN']=df['Admin_Code_CHN'].astype(np.float64)
df['Lon']=df['Lon'].astype(np.float64)
df['Lat']=df['Lat'].astype(np.float64)
df['RHU']=df['RHU'].astype(np.float64)
df['TEM']=df['TEM'].astype(np.float64)
df['Datetime']=pd.to_datetime(df['Datetime'])


####################### TEMPER
data=df[(df['TEM']<50.0)&(df['TEM']>-20.0)]
lon,lat,temp,alti = data['Lon'],data['Lat'],data['TEM'],data['Alti']
DZdf=data[data['Admin_Code_CHN'].isin([511701,511702,511703,511722,511723,511724,511725,511781])]

#########样本集X自变量，temp因变量，分为训练集和测试集
X=data.loc[:,('Lon','Lat','Alti')]  #自变量为3个因子
X_train,X_test, y_train, y_test = train_test_split(X,temp,test_size = 0.1,random_state=0)
#print ('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,y_train.shape, X_test.shape,y_test.shape))
linreg = LinearRegression()
model=linreg.fit(X_train, y_train)
#print (model)# 训练后模型截距
#print (linreg.intercept_)# 训练后模型权重（特征个数无变化）
#print (linreg.coef_)
#linreg.score(X_train,y_train)
b=linreg.coef_[0]
c=linreg.coef_[1]
d=linreg.coef_[2]
T0=linreg.intercept_
#T=T0 + a*Lon + b*Lat + c*Alti
#########利用DEM数据进行回归算出范围内的温度分布，同时这里也声明后续使用的经纬度信息
lons=np.linspace(106,109,3600)  #这里的3600是有DEM数据决定的，90m也就是3″分辨率的DEM，这个范围内就是3600个点
lats=np.linspace(30,33,3600) 
olon,olat = np.meshgrid(lons,lats)
T=T0+ b*olon+ c*olat + d*tif16[::-1] #因为DEM数据是从北到南的，所以这里加一个反序
#T=np.zeros((3600,3600), dtype=float)
#for i in range(0,3600):
#    for j in range(0,3600):
#        T[i][j]=T0+ b*lons[i] + c*lats[j] + d*tif16[i][j]

################绘图部分
plt.ion()
proj=ccrs.PlateCarree()
fig = plt.figure(figsize=(16,9.6),facecolor='#666666',edgecolor='Blue',frameon=False)
ax = fig.add_subplot(111,projection=proj)
ax.spines['top'].set_visible(False) #去掉上边框
ax.spines['bottom'].set_visible(False) #去掉下边框
ax.spines['left'].set_visible(False) #去掉左边框
ax.spines['right'].set_visible(False) #去掉右边框
#shpinfo=m.readshapefile('data/dzx','达州市',color='gray',linewidth=1)
clevs=np.arange(-11,42,1)#自定义颜色列表
# cdict=['#0200FD','#023EBF','#027887','#02916E','#00BB42','#00D32C','#00F30C','#4CFF00','#66FF00','#90FF02','#B7FF00','#FDFF02',
#       '#F1EB02','#FFC100','#FFAB00','#F5A104','#FF8900','#FF6E00','#FF4C00','#FF2A00','#E50012','#C100FF']#自定义颜色列表
# my_cmap=colors.ListedColormap(cdict)#自定义颜色映射color-map
# norm = mpl.colors.BoundaryNorm(clevs,my_cmap.N)#基于离散区间生成颜色映射索引
norm = mpl.colors.Normalize(vmin=20, vmax=30)
cf = ax.contourf(olon,olat,T,clevs,cmap='hsv_r',norm=norm,alpha=0.7)
#添加达州区内的站点填值
for i in DZdf.index:
    plt.text(DZdf['Lon'][i],DZdf['Lat'][i],round(DZdf['TEM'][i],1),va='center',fontsize=7)
#裁切
clip=maskout.shp2clip(cf,ax,'../shp/dz')
plt.show()
# plt.savefig('dzjiance/TEMPER/'+a[0:8]+'/'+a+'.png',format='png', transparent=True, bbox_inches='tight',dpi=300)
# plt.close()