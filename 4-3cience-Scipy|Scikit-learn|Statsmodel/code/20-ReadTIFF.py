# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:03:03 2019

@author: Administrator
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
###from PIL import Image

filename=r'..\shp\srtm_58_06_Clip1_1.tif'
tif16 = imageio.imread(filename)
###tif16 = Image.open(filename) 
x=np.linspace(106,109,3600)
y=np.linspace(33,30,3600)

fig=plt.figure()
ax=fig.add_subplot(111)

# ax.contourf(x,y,tif16)
ax.imshow(tif16)

