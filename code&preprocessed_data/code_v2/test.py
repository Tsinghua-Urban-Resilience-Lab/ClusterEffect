# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:35:33 2023

@author: admin
"""

#random_explore & explore_in_cluster（随机探索的比例）
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import pandas as pd
import os  # 补上了漏掉的 os 库

pnew_random = np.empty(shape=[0,3])
path = r'D:\results\zxy\new_result\guangzhou\cluster_feature'
filenames = os.listdir(path)
num = 0

#S;探索概率；ID
pnew_cluster = np.empty(shape=[0,3])
for filename in filenames:
    print(num)
    sub_path=r'D:\results\zxy\new_result\guangzhou\\pattern'+'\\'+filename
    #,arrive_time,arrive_day,arrive_hour,leave_time,leave_day,leave_hour,cluster,lon,lat,location
    data=pd.read_csv(sub_path, sep=',', header=0)
    a=list(set(list(data['cluster'])))
    l=list(set(list(data['location'])))
    if len(a)<=1:
        continue
    #访问地点过少没有聚类结果的个体
    s0=np.zeros((len(l)+1,2))
    x=[[data['lon'][0],data['lat'][0]]]
    for i in range(1,len(data)):
        if len(x)<2:
            if [data['lon'][i],data['lat'][i]] not in x:
                x+=[[data['lon'][i],data['lat'][i]]]
            continue
        else:
            k=len(x)

            X=np.array(x)
            hdb = hdbscan.HDBSCAN(min_cluster_size=2,cluster_selection_epsilon=0.01,prediction_data=True).fit(X)
            labels=hdb.labels_
            if labels[-1]==-1:
                s0[k,1]+=1
                s0[k,0]+=1
            else:
                s0[k,1]+=1
    num+=1 

# ==================== 结果保存部分 ====================
output_dir = r'D:\results\zxy\new_result\code&preprocessed_data\data'

# 方案 A：分开保存为两个 .npy 文件
np.save(os.path.join(output_dir, 'pnew_random2.npy'), pnew_random)
print("数据已成功保存为 .npy 格式！")

