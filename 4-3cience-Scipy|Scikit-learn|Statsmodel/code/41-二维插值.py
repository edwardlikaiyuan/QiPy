# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:40:03 2019

@author: Administrator
"""
# Copyright (c) 2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Point Interpolation
===================

Compares different point interpolation approaches.
"""
#科学工具
import numpy as np
#可视化工具
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
#大气科学工具
from metpy.interpolate import (interpolate_to_grid, remove_nan_observations,
                               remove_repeat_coordinates)
#地图、地理信息工具
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import maskout


################# 画布 + (投影ed)绘图区 ##################
def basic_map(proj, title):
    """Make our basic default map for plotting"""
    fig = plt.figure(figsize=(7, 6))
    # add_metpy_logo(fig, 0, 80, size='large')
    view = fig.add_axes([0.05, 0.05, 0.85, 0.9], projection=proj)
    view.set_title(title)
    view.set_extent([118,126,38,44])
    # view.add_feature(cfeature.STATES.with_scale('50m'))
    # view.add_feature(cfeature.OCEAN)
    # view.add_feature(cfeature.COASTLINE)
    # view.add_feature(cfeature.BORDERS, linestyle=':')
    return fig, view

################# 数据准备（文件读取 → 坐标转换） ##################
def station_test_data(variable_names, proj_from=None, proj_to=None):
    all_data = np.loadtxt('../data/Rain19052708.000', skiprows=14,usecols=(0,1,2,4),
                              dtype=np.dtype([('stid', '5S'), ('lon', 'f'), ('lat', 'f'),('rain', 'f')]))
    all_stids = [s.decode('ascii') for s in all_data['stid']]
    data = np.concatenate([all_data[all_stids.index(site)].reshape(1, ) for site in all_stids])
    value = data[variable_names]
    lon = data['lon']
    lat = data['lat']
    # from经纬度 → to目标投影下的地球坐标
    if proj_from is not None and proj_to is not None:
        try:
            proj_points = proj_to.transform_points(proj_from, lon, lat)
            return proj_points[:, 0], proj_points[:, 1], value
        except Exception as e:
            print(e)
            return None
    return lon, lat, value



################# 开始绘图 ##################
from_proj = ccrs.PlateCarree()
to_proj = ccrs.LambertConformal(central_longitude=110.0000, central_latitude=38.0000)
# to_proj = ccrs.AlbersEqualArea(central_longitude=110.0000, central_latitude=38.0000)

x, y, temp = station_test_data('rain', from_proj, to_proj)
x, y, temp = remove_nan_observations(x, y, temp)
x, y, temp = remove_repeat_coordinates(x, y, temp)
# # #色彩定制：24小时降水量级色标
levels = [0.1,10.,25.,50.,100.,250.,500]#自定义分级
cdict = ['#A6F28F', '#3DBA3D', '#61B8FF', '#0000FF', '#FA00FA', '#800040']#自定义颜色列表
cmap=mpl.colors.ListedColormap(cdict)#自定义颜色映射colormap
norm = mpl.colors.BoundaryNorm(levels, cmap.N, clip=True)#基于离散区间生成颜色映射索引

################ 插值方法选择 #########################
# # Scipy的线性插值Scipy.interpolate linear
# # ------------------------
gx, gy, img = interpolate_to_grid(x, y, temp, interp_type='linear', hres=1000)
img = np.ma.masked_where(np.isnan(img), img)
fig, view = basic_map(to_proj, 'Scipy Linear')
mmb = view.contourf(gx, gy, img, levels, cmap=cmap, norm=norm)
fig.colorbar(mmb, shrink=.4, pad=0, boundaries=levels)

###########################################
# # Metpy的自然相邻插值 Natural neighbor interpolation (MetPy implementation)
# # -----------------------------------------------------
# # `Reference <https://github.com/Unidata/MetPy/files/138653/cwp-657.pdf>`_
# gx, gy, img = interpolate_to_grid(x, y, temp, interp_type='natural_neighbor', hres=75000)
# img = np.ma.masked_where(np.isnan(img), img)
# fig, view = basic_map(to_proj, 'Metpy Natural Neighbor')
# mmb = view.contourf(gx, gy, img, levels, cmap=cmap, norm=norm)
# # mmb = view.pcolormesh(gx, gy, img,cmap=cmap, norm=norm)
# fig.colorbar(mmb, shrink=.4, pad=0, boundaries=levels)

###########################################
# Metpy的克莱斯曼插值  Cressman interpolation
# ----------------------
# search_radius = 100 km
#
# grid resolution = 25 km
#
# min_neighbors = 1
# gx, gy, img = interpolate_to_grid(x, y, temp, interp_type='cressman', minimum_neighbors=1,
#                                   hres=5000, search_radius=100000)
# img = np.ma.masked_where(np.isnan(img), img)
# fig, view = basic_map(to_proj, 'Metpy Cressman')
# mmb = view.contourf(gx, gy, img, levels, cmap=cmap, norm=norm)
# # mmb = view.pcolormesh(gx, gy, img,cmap=cmap, norm=norm)
# fig.colorbar(mmb, shrink=.4, pad=0, boundaries=levels)

# ###########################################
# # Barnes Interpolation
# # --------------------
# # search_radius = 100km
# #
# # min_neighbors = 3
# gx, gy, img1 = interpolate_to_grid(x, y, temp, interp_type='barnes', hres=5000,
#                                     search_radius=100000)
# img1 = np.ma.masked_where(np.isnan(img1), img1)
# fig, view = basic_map(to_proj, 'Metpy Barnes')
# mmb = view.contourf(gx, gy, img1, levels, cmap=cmap, norm=norm)
# # mmb = view.pcolormesh(gx, gy, img1, cmap=cmap, norm=norm)
# fig.colorbar(mmb, shrink=.4, pad=0, boundaries=levels)

# ###########################################
# # 径向基函数 Radial basis function interpolation
# # ------------------------------------
# # linear
# gx, gy, img = interpolate_to_grid(x, y, temp, interp_type='rbf', hres=10000, rbf_func='cubic',
#                                   rbf_smooth=0)
# img = np.ma.masked_where(np.isnan(img), img)
# fig, view = basic_map(to_proj, 'Radial Basis Function')
# mmb = view.contourf(gx, gy, img, levels, cmap=cmap, norm=norm)
# # mmb = view.pcolormesh(gx, gy, img, levels,cmap=cmap, norm=norm)
# fig.colorbar(mmb, shrink=.4, pad=0, boundaries=levels)

################# 地理信息 地图裁切 ##################
# # #从全国市级地图中获取辽宁省的市级边界并加载
shpname='../shp/City.shp'
adm1_shapes=list(shpreader.Reader(shpname).geometries())
view.add_geometries(adm1_shapes[36:50],ccrs.PlateCarree(),edgecolor='k',facecolor='None',linewidth=0.1)    #36:72东三省 
# # #裁切
clip=maskout.shp2clip(mmb,view,'../shp/liaoning','辽宁省')

## 显示输出
plt.show()




