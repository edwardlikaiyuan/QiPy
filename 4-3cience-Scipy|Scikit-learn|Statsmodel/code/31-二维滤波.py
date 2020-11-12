# 引用部分
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# 数据准备
df = pd.read_csv('../data/ET_海平面气压(hpa)分析.txt',skiprows=6,sep='\s+')
df = df[::-1]
###二维contour的滤波平滑
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# # 二维contour的滤波平滑  -  高斯滤波
# # Increase the value of sigma to increase the amount of blurring.
# # order=0 means gaussian kernel
# Z2 = ndimage.gaussian_filter(df, sigma=5.0, order=0)
# fig=plt.figure()
# ax=fig.add_subplot(1,2,1)
# ax.contour(df,levels=np.arange(980,1040,1))
# ax=fig.add_subplot(1,2,2)
# ax.contour(Z2,levels=np.arange(980,1040,1))
# plt.show()






# #Smoothing of a 2D signal
# #Convolving a noisy image with a gaussian kernel
# #sg.convolved的二维拓展
# def gauss_kern(size, sizey=None):
#     """ Returns a normalized 2D gauss kernel array for convolutions """
#     size = int(size)
#     if not sizey:
#         sizey = size
#     else:
#         sizey = int(sizey)
#     x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
#     g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
#     return g / g.sum()

# def blur_image(im, n, ny=None) :
#     """ blurs the image by convolving with a gaussian kernel of typical
#         size n. The optional keyword argument ny allows for a different
#         size in the y direction.
#     """
#     g = gauss_kern(n, sizey=ny)
#     # print(type(g),g.shape)
#     # print(g)
#     improc = sg.convolve(im,g, mode='valid')
#     return(improc)

# z = blur_image(df, 15)
# # filter_kernel = np.array([[-1,-1,-1],
# #                     [-1,8,-1],
# #                     [-1,-1,-1]])
# # z = sg.convolve2d(df, filter_kernel, boundary='fill', fillvalue=0)
# plt.contour(df)
# plt.contour(z,colors='k')


# # scipy.ndimage
# # https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html
# # 相关与卷积
# correlate1d
# correlate 
# convolve1d 
# convolve 
# # 平滑滤波器
# gaussian_filter1d
# gaussian_filter
# uniform_filter1d
# uniform_filter 
# # 基于订单统计的过滤器
# minimum_filter1d 
# maximum_filter1d 
# minimum_filter
# maximum_filter 
# rank_filter 
# percentile_filter
# median_filter 
# # 衍生工具
# prewitt
# sobel
# generic_laplace
# gaussian_laplace
# generic_gradient_magnitude
# # 通用筛选器函数
# generic_filter1d 
# generic_filter
# # 傅里叶域滤波器
# fourier_shift 
# fourier_gaussian
# fourier_uniform
# fourier_ellipsoid



# # scipy.ndimage
# 插值的方法
# ·Spline pre-filters
# spline_filter1d
# spline_filter
# geometric_transform
# affine_transform 
# shift
# zoom 
# rotate
































# import scipy
# # Two dimensional data smoothing and least-square gradient estimate
# #!python numbers=enable
# def sgolay2d ( z, window_size, order, derivative=None):
#     """
#     """
#     # number of terms in the polynomial expression
#     n_terms = ( order + 1 ) * ( order + 2)  / 2.0

#     if  window_size % 2 == 0:
#         raise ValueError('window_size must be odd')

#     if window_size**2 < n_terms:
#         raise ValueError('order is too high for the window size')

#     half_size = window_size // 2

#     # exponents of the polynomial. 
#     # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
#     # this line gives a list of two item tuple. Each tuple contains 
#     # the exponents of the k-th term. First element of tuple is for x
#     # second element for y.
#     # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
#     exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

#     # coordinates of points
#     ind = np.arange(-half_size, half_size+1, dtype=np.float64)
#     dx = np.repeat( ind, window_size )
#     dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

#     # build matrix of system of equation
#     A = np.empty( (window_size**2, len(exps)) )
#     for i, exp in enumerate( exps ):
#         A[:,i] = (dx**exp[0]) * (dy**exp[1])

#     # pad input array with appropriate values at the four borders
#     new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
#     Z = np.zeros( (new_shape) )
#     # top band
#     band = z[0, :]
#     Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
#     # bottom band
#     band = z[-1, :]
#     Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
#     # left band
#     band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
#     Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
#     # right band
#     band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
#     Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
#     # central band
#     Z[half_size:-half_size, half_size:-half_size] = z

#     # top left corner
#     band = z[0,0]
#     Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
#     # bottom right corner
#     band = z[-1,-1]
#     Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

#     # top right corner
#     band = Z[half_size,-half_size:]
#     Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
#     # bottom left corner
#     band = Z[-half_size:,half_size].reshape(-1,1)
#     Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

#     # solve system and convolve
#     if derivative == None:
#         m = np.linalg.pinv(A)[0].reshape((window_size, -1))
#         return scipy.signal.fftconvolve(Z, m, mode='valid')
#     elif derivative == 'col':
#         c = np.linalg.pinv(A)[1].reshape((window_size, -1))
#         return scipy.signal.fftconvolve(Z, -c, mode='valid')
#     elif derivative == 'row':
#         r = np.linalg.pinv(A)[2].reshape((window_size, -1))
#         return scipy.signal.fftconvolve(Z, -r, mode='valid')
#     elif derivative == 'both':
#         c = np.linalg.pinv(A)[1].reshape((window_size, -1))
#         r = np.linalg.pinv(A)[2].reshape((window_size, -1))
#         return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')
    
# #  二维的Savitsky-Golay滤波
# Zf = sgolay2d(df.values, window_size=29, order=4)
# plt.contour(df)
# plt.contour(Zf)








































# # # 1-D interpolation一维插值  scipy.interpolate  必须过点！！！
# # 用观测点建模，以反映更为连续的插值节点上的情况
# # from scipy.interpolate import interp1d
# # f = interp1d(x, y, kind='cubic')
# # # f2 = interp1d(x, y, kind='cubic')
# # xnew = np.linspace(0, len(df)-1, num=len(df)*2, endpoint=False)
# # plt.plot(x, y, 'o', xnew, f(xnew))#, '-', xnew, f2(xnew), '--')
# # plt.show()