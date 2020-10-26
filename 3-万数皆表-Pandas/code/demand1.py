# 需求站点列表：从txt文件中读取需要站点站号，列表
with open('../data/站点信息.txt') as f:
    lines=f.read().split(sep='\n')
    stidls=[int(line) for line in lines]  #s=list(map(int, lines))
# 获得原始数据文件列表，用于遍历
import glob
pattern='../data/'+'2019*.csv'
files =[]
for file in glob.glob(pattern):
    files.append(file)
# 主循环：（读取数据→筛选站点→重塑一行→插入时间）*循环追加 → 输出
import pandas as pd
out=pd.DataFrame()#输出结果的dataframe
# for filename in files:
for filename in glob.glob(pattern):
    import chardet
    with open(filename, 'rb') as f:
        coding=str(chardet.detect(f.readline())['encoding'])
    time = filename[-16:-4]
    df = pd.read_csv(filename,usecols=[0,3,4],encoding=coding)  # df = df[['站号','风向','风速']]
    data = df[df['站号'].isin(stidls)].sort_values(by='站号').reset_index(drop=True) #如果缺测……
    data = pd.DataFrame(data.drop('站号',axis=1).values.reshape(1,20),columns=['风向','风速']*10) # data = data.drop('站号',axis=1).stack().reset_index(drop=True)
    data.insert(0,'时间',time) 
    out = out.append(data)#追加到out当中，最后再输出
out.to_csv('result.csv',index=False,header=True,encoding='gbk')






# filename = '../data/201904161001.csv'
# time = filename[-16:-4]
# df = pd.read_csv(filename,usecols=[0,6,7])  # df = df[['站号','风向','风速']]
# data = df[df['站号'].isin(stidls)].sort_values(by='站号').reset_index(drop=True)
# data = data.drop('站号',axis=1).stack().reset_index(drop=True)
# data = pd.Series(time).append(data).reset_index(drop=True)


# filename = '../data/201904161005.csv'
# time = filename[-16:-4]
# df1 = pd.read_csv(filename,usecols=[0,6,7])  # df = df[['站号','风向','风速']]
# data1 = df1[df1['站号'].isin(stidls)].sort_values(by='站号').reset_index(drop=True)
# data1 = data1.drop('站号',axis=1).stack().reset_index(drop=True)
# data1 = pd.Series(time).append(data1).reset_index(drop=True)

# data=pd.concat([data,data1],axis=1).T
# data.to_csv('abc.csv')
