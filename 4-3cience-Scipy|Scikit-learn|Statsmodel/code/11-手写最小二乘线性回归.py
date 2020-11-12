import numpy as np
import matplotlib.pyplot as plt
 
class SimpleRegress(object):
    def __init__(self, x_data, y_data):
 
        self.x_data = x_data
        self.y_data = y_data
        self.b0 = 0
        self.b1 = 1
 
        return
 
    def calculate_work(self):       # 回归方程中b0、b1的求解
 
        x_mean = np.mean(self.x_data)   # x_mean= 14.0
        y_mean = np.mean(self.y_data)   # y_mean= 130.0
        x1 = self.x_data - x_mean   # x1= [-12.  -8.  -6.  -6.  -2.   2.   6.   6.   8.  12.]
        y1 = self.y_data - y_mean   # y1= [-72. -25. -42. -12. -13.   7.  27.  39.  19.  72.]
        s = x1 * y1     # s= [864. 200. 252.  72.  26.  14. 162. 234. 152. 864.]
        u = x1 * x1     # u= [144.  64.  36.  36.   4.   4.  36.  36.  64. 144.]
        self.b1 = np.sum(s) / np.sum(u)      # b1= 5.0
        self.b0 = y_mean - self.b1 * x_mean       # b0= 60.0
 
        return
 
    def test_data_work(self, text_data):    # 回归方程的建立与数值预测
 
        result = list([])
        for one_test in text_data:
            y = self.b0 + self.b1 * one_test
            result.append(y)
        return result
 
    def root_data_view(self):    # 绘制源数据可视化图
        plt.scatter(x_data, y_data, label='simple regress', color='k', s=5)  # s 点的大小
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        return
 
    def test_data_view(self):    # 绘制回归线
        # 绘制回归线两个点的数据
        x_min = np.min(self.x_data)
        x_max = np.max(self.x_data)
        y_min = np.min(self.y_data)
        y_max = np.max(self.y_data)
        x_plot = list([x_min, x_max])
        y_plot = list([y_min, y_max])
        # 绘制
        plt.scatter(x_data, y_data, label='root data', color='b', s=15)  # s 点的大小
        plt.scatter(test_data, result, label='test data', color='r', s=15)  # s 点的大小
        plt.plot(x_plot, y_plot, label='regression line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('simple linear regression')
        plt.show()
        return
 
x_data = list([2, 6, 8, 8, 12, 16, 20, 20, 22, 26])
y_data = list([58, 105, 88, 118, 117, 137, 157, 169, 149, 202])
test_data = list([16])
 
sr = SimpleRegress(x_data, y_data)
sr.calculate_work()
result = sr.test_data_work(test_data)       # result= [140.0]
sr.test_data_view()