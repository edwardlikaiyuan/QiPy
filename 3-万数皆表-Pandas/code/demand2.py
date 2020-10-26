import pandas as pd
a=r"../data/Surf20092308.000"
#a=sys.argv[1]
#a=r'x:\surface\plot\19112420.000'
year='20'+a[-12:-10]
month=a[-10:-8]
day=a[-8:-6]
valid=a[-6:-4]


df=pd.read_csv(a,skiprows=3,header=None,encoding="GB2312",sep='\s+').drop([0])
df1=df.iloc[::2].reset_index(drop=True)
df2=df.iloc[1::2].reset_index(drop=True)
df1=df1.drop(columns=[12,13])
res=pd.concat([df1,df2],axis=1,ignore_index=True)
res.columns=["stid","lon","lat","h","lev","cloud_fraction","wind_dir",
      "wind_speed","slp","p3","w1","w2","r6","lc","lcc","lch",
      "dew_point_temperature","vv","weather","air_temperature","mc","hc","s1","s2","T24","P24"]
res[['stid','cloud_fraction','weather','w1','w2','lcc','lc','lch','mc','hc','p3','r6']]=res[['stid','cloud_fraction','weather','w1','w2','lcc','lc','lch','mc','hc','p3','r6']].astype('int')
res=res[res['stid']<80000]
res['weather']=res['weather'].replace(9999,0)
res['wind_speed']=res['wind_speed'].replace(9999,0)
res['weather']=res['weather'].astype('int')
res['cloud_fraction']=res['cloud_fraction'].replace(9999,10)