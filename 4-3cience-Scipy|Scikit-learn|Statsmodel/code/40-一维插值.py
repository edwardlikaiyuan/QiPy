# 引用部分
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.colors as clrs


# # # #一维插值
# # 数据准备
# df = pd.read_excel('../data/北京虚拟温度.xlsx')
# x = np.arange(0,24)
# y = df['温度'][0:24].values


# # 1-D interpolation一维插值  scipy.interpolate  必须过点！！！

'''# # 用观测点建模，以反映更为连续的插值节点上的情况
# 将插值类型指定为字符串（
# “linear”、“nearest”、“zero”、“slinear”、“quadratic”、“cubic”、“previous”、“next”
# 其中“zero”、“slinear”、“quadratic”和“cubic”表示零阶、一阶、二阶或三阶的样条曲线插值；
# “previous”和“next”只返回点的上一个或下一个值）或指定要使用的样条曲线插值器顺序的整数。
# 默认值为“线性”。'''
# from scipy.interpolate import interp1d
# f = interp1d(x, y, kind='cubic')#三次多项式插值
# f2 = interp1d(x, y, kind='linear')#线性插值
# xnew = np.linspace(0, 24-1, num=24*5, endpoint=True)#新的插值节点

# plt.plot(xnew, f(xnew),'k-h')
# plt.plot(xnew, f2(xnew),'b:')
# plt.plot(x, y, 'ro',markersize=15)# #原始温度时间序列/信号
# plt.show()

# # 样条插值实现，默认是三次样条
# tck = scipy.interpolate.splrep(x,y,s=0)
# xnew = np.linspace(0, 24-1, num=24*5, endpoint=True)#新的插值节点
# ynew = scipy.interpolate.splev(xnew, tck, der=0)
# plt.plot(x, y, 'x', xnew, ynew)


# # 径向基函数一维插值
# from scipy.interpolate import Rbf
# rbf = Rbf(x, y)
# xnew = np.linspace(0, 24-1, num=24*5, endpoint=True)#新的插值节点
# fnew = rbf(xnew)
# plt.plot(x, y, 'x',     xnew, fnew, 'g')
