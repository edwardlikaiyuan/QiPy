# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:51:26 2020

@author: OLDLee
"""

import glob
import pandas as pd
pattern='../data/'+'RAIN*.csv'
dfRain = pd.DataFrame(columns=['站号','经度','纬度','海拔高度','观测值'])
for filename in glob.glob(pattern):
    df=pd.read_csv(filename)
    # print(df.shape)
    df=df[['站号','经度','纬度','海拔高度','观测值']]
    dfRain=dfRain.append(df, ignore_index=True)
cumulative=dfRain.groupby('站号').sum().sort_values(by='站号')