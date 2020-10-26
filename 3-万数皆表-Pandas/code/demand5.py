# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:51:26 2020

@author: OLDLee
"""

import pandas as pd
df=pd.read_excel('../data/timeSeries.xlsx',sheet_name='DataAll')
   
# dftime=df['obsTime'][::14].reset_index(drop=True)
dftime =df['obsTime'].drop_duplicates().reset_index(drop=True)