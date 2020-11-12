# 引用部分
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# 数据准备
df = pd.read_excel('../data/CAPE-WMAX.xlsx')
df =df.sort_values(by='CAPE')
cape,wmax=df['CAPE'],df['WMAX']

#=============================================================================
# # # numpy的回归方法：np.polyfit()最小二乘多项式拟合函数  np.linalg.lstsq()scipy.polyfit()
# '''
# 这是一个非常一般的最小二乘多项式拟合函数，
# 它适用于任何 degree 的数据集与多项式函数（具体由用户来指定），
# 其返回值是一个（最小化方差）回归系数的数组。
# 对于简单的线性回归而言，你可以把 degree 设为 1。
# 如果你想拟合一个 degree 更高的模型，你也可以通过从线性特征数据中建立多项式特征来完成。
# '''
# fit_func=np.polyfit(cape,wmax,1)
# xnew=np.linspace(cape.min(),cape.max(),3000)
# ynew=np.polyval(fit_func,xnew)
# p1 = np.poly1d(fit_func)
# plt.plot(xnew,ynew,'k-.',label=p1)
# plt.legend()
# plt.title('np.polyfit()')

#=============================================================================
# # # scipy.stas 的线性回归方法linregress
# '''
# 这是 Scipy 中的统计模块中的一个高度专门化的线性回归函数。
# 其灵活性相当受限，因为它只对计算两组测量值的最小二乘回归进行优化。
# 因此，你不能用它拟合一般的线性模型，或者是用它来进行多变量回归分析。
# 但是，由于该函数的目的是为了执行专门的任务，
# 所以当我们遇到简单的线性回归分析时，这是最快速的方法之一。
# 除了已拟合的系数和截距项（intercept term）外，
# 它还会返回基本的统计学值如 R 系数与标准差。
# '''
# import scipy.stats as st
# # 线性拟合，可以返回斜率，截距，r 值，p 值，标准误差
# slope, intercept, r_value, p_value, std_err = st.linregress(cape, wmax)
# print(slope)# 输出斜率
# print(intercept) # 输出截距
# print(r_value**2) # 输出 r^2
# xnew=np.linspace(cape.min(),cape.max(),3000)
# ynew=slope*xnew +intercept
# plt.plot(xnew,ynew,'y-.',zorder=1,label=('y= %.3f*x + %.3f' % (slope,intercept)))
# plt.legend()
# plt.title('scipy.stats.linregress()')

#=============================================================================
# # # scipy.optimize 的拟合函数 curve_fit
# '''
# 这个方法与 Polyfit 方法类似，但是从根本来讲更为普遍。
# 通过进行最小二乘极小化，这个来自 scipy.optimize模块的
# 强大函数可以通过最小二乘方法将用户定义的任何函数拟合到数据集上。
# 对于简单的线性回归任务，我们可以写一个线性函数：mx+c，我们将它称为估计器。
# 它也适用于多变量回归。它会返回一个由函数参数组成的数列，
# 这些参数是使最小二乘值最小化的参数，以及相关协方差矩阵的参数。
# '''
# from scipy.optimize import curve_fit    #引用
# def func1(x,a,b):                      #定义目标函数
#     return a*x + b                    # 目标函数表达式
# def func2(x,a,b,c):                      #定义目标函数
#     return a*x + b * x**2 +c          # 目标函数表达式
# def func3(x,a,b,c):                      #定义目标函数
#     return a*np.sqrt(b*x)+c             # 目标函数表达式
# popt1, pcov1 = curve_fit(func1,cape,wmax)  #拟合：opt拟合参数也就是a,b,c ，pcov参数的协方差矩阵
# popt2, pcov2 = curve_fit(func2,cape,wmax)  #拟合：opt拟合参数也就是a,b,c ，pcov参数的协方差矩阵
# popt3, pcov3 = curve_fit(func3,cape,wmax)  #拟合：opt拟合参数也就是a,b,c ，pcov参数的协方差矩阵
# plt.plot(cape, func1(cape, *popt1), 'r-',label=('$y= %.3f*x + %.3f$' % tuple(popt1)))
# plt.plot(cape, func2(cape, *popt2), 'g--',label=('$y= %.3f*x %.3f*x^2 + %.3f$' % tuple(popt2)))
# plt.plot(cape, func3(cape, *popt3), 'b-.',label=('$W_{max}=%.3f x \sqrt{%.3f*CAPE} %.3f$' % tuple(popt3)))
# plt.legend()
# plt.title('scipy.optimize.curve_fit()')

#=============================================================================
# statsmodels的回归方法：ols.fit()线性回归  
# '''
# 对于线性回归，人们可以从这个包调用 OLS 或者是 
# Ordinary least squares 函数来得出估计过程的最终统计数据。
# 需要记住的一个小窍门是，必须要手动为数据 x 添加一个常数，以用于计算截距。
# 否则，只会默认输出回归系数。下方表格汇总了 OLS 模型全部的结果。
# 它和任何函数统计语言（如 R 和 Julia）一样丰富。
# '''
# from statsmodels.formula.api import ols
# df=pd.read_excel('../data/CAPE-WMAX.xlsx')
# cape,wmax=df['CAPE'],df['WMAX']
# lm=ols('WMAX~CAPE',df).fit() # 构建最小二乘模型并拟合，将自变量与因变量用统计语言公式表示，将全部数据导入
# print(lm.summary())
# print(lm.params)
# plt.plot(df['CAPE'],lm.fittedvalues,'r',linewidth=2,zorder=0,label=('y= %.3f*x + %.3f' % tuple(lm.params[::-1])))
# plt.legend()
# plt.title('statsmodels.formula.ols.fit()')

# # # 多元线性回归
# # import statsmodels.api as sm
# # datas = pd.read_excel('../data/CAPE-WMAX.xlsx') # 读取 excel 数据
# # y = datas.iloc[:, 1] # 因变量为第 2 列数据
# # x = datas.iloc[:, 2:6] # 自变量为第 3 列到第 6 列数据
# # x = sm.add_constant(x) # 若模型中有截距，必须有这一步
# # model = sm.OLS(y, x).fit() # 构建最小二乘模型并拟合
# # print(model.summary()) # 输出回归结果



#=============================================================================
# # # scikit-learn的回归方法：LinearRegression
# '''
# 这个方法经常被大部分机器学习工程师与数据科学家使用。
# 然而，对于真实世界的问题，它的使用范围可能没那么广，
# 我们可以用交叉验证与正则化算法比如 Lasso 回归和 Ridge 回归来代替它。
# 但是要知道，那些高级函数的本质核心还是从属于这个模型。
# '''
# from sklearn import linear_model
# x = cape[:, np.newaxis]
# y = wmax[:, np.newaxis]
# reg = linear_model.LinearRegression()
# reg.fit(x,y)
# predicts = reg.predict(x) # 预测值
# R2 = reg.score(x, y) # 拟合程度 R2
# print('R2 = %.2f' % R2) # 输出 R2
# print(reg.coef_, reg.intercept_) # 输出斜率和截距
# plt.plot(x, predicts, color = 'red', label='y= %.3f*x + %.3f' % (reg.coef_, reg.intercept_))
# plt.title('sklearn.linear_model.LinearRegression()')
# plt.legend()

norm1=clrs.Normalize(vmin=0,vmax=2000)
plt.scatter(cape,wmax,marker='^',c=cape,norm=norm1,cmap='rainbow',s=wmax,alpha=0.75,
            linewidths=wmax*0.01,edgecolors='orange')
plt.xlabel('CAPE')
plt.ylabel('Wmax')
plt.show()
