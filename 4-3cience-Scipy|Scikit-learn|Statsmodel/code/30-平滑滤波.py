# 引用部分
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# 数据准备
df = pd.read_excel('../data/北京虚拟温度.xlsx')
x = np.arange(len(df))
y = df['温度'].values

#原始温度时间序列/信号
plt.plot(y,'k',lw=0.5)


#=======================================================
# 一维中值滤波           最大值、最小值、均值滤波
# tmp_smooth = sg.medfilt(y,25) #中值滤波 cv2.medianBlur(y , 73)# 均值滤波cv2.blur   GaussianBlur
# plt.plot(tmp_smooth,'r',lw=2)


# #=======================================================
# # 卷积滤波。用一个平均权重的滤波器卷积，就是滑动平均
# # dataframe的rolling方法
# z1 = sg.convolve(y, np.ones(169)/169) #[0.3,0.5,0.2]
# plt.plot(z1,'r',lw=4)
# plt.plot(df['温度'].rolling(window=169).mean(),'y',lw=1)
# plt.xticks(np.arange(15*24,len(y),30*24))
# plt.gca().xaxis.set_ticklabels(np.arange(1,13)) #np.append([''],np.arange(1,13)))
# plt.show()


# import scipy.ndimage as ni
# z1 = ni.uniform_filter1d(y,169)#此滤波器的作用是取每个像素及其相邻像素的算术平均值。大小是计算算术平均值的子数组的大小。没有足够邻居的像素的标准是要反映出来的。
# plt.plot(z1,'r',lw=4)
# ynew = ni.convolve1d(y,np.ones(169)/169)#一维卷积
# plt.plot(z1,'b',lw=1)


#=======================================================
# # scipy.signal.wiener维纳滤波
# z1 = sg.wiener(y,1609)
# plt.plot(z1,'b',lw=3)



#=======================================================
# # Savitzky-Golay滤波
# '''
# （y , m ,n）  m窗口宽度 ，n多项式次数
# 对2m+1内的数据进行n阶多项式拟合，最小二乘……
# 一维滤波器
# Savitzky-Golay滤波器是一种数字滤波器、平滑算法、卷积平滑
# 它可以平滑一组数据，即在不改变信号趋势的情况下提高数据的精度。
# 通过卷积的过程实现，即通过线性最小二乘法将相邻数据点的连续子集与一个低次多项式拟合。
# 当数据点的间距相等时，可以找到最小二乘方程的解析解，
# 其形式是一组可以应用于所有数据子的“卷积系数”
# '''
# tmp_smooth = sg.savgol_filter(y,169,5)
# plt.plot(tmp_smooth,'g',lw=3)





#=======================================================
# scipy.ndimage.gaussian_filter1d(input, sigma, axis=- 1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#mode{‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
# import scipy.ndimage as ni
# ynew = ni.gaussian_filter1d(y,sigma=102.0,order=0,mode='nearest')
# ynew = ni.uniform_filter1d(y,169)#此滤波器的作用是取每个像素及其相邻像素的算术平均值。大小是计算算术平均值的子数组的大小。没有足够邻居的像素的标准是要反映出来的。
# plt.plot(ynew,'r',lw=4)
# ynew = ni.convolve1d(y,np.ones(169)/169)#一维卷积
# plt.plot(ynew,'b',lw=1)



























# # 1-D interpolation一维插值  scipy.interpolate  必须过点！！！
# 用观测点建模，以反映更为连续的插值节点上的情况
# from scipy.interpolate import interp1d
# f = interp1d(x, y, kind='cubic')
# # f2 = interp1d(x, y, kind='cubic')
# xnew = np.linspace(0, len(df)-1, num=len(df)*2, endpoint=False)
# plt.plot(x, y, 'o', xnew, f(xnew))#, '-', xnew, f2(xnew), '--')
# plt.show()

