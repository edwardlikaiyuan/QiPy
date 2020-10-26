import numpy as np
import pandas as pd
fileName='../data/Vor20060808.000'
df=pd.read_csv(fileName, skiprows=4, header=None, encoding="GB2312", sep='\s+')
vor=df.values.reshape(29,60)
vor=np.delete(vor,list(range(53,60)),axis=1)