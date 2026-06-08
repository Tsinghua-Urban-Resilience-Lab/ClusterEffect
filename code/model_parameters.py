# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:35:33 2023

@author: admin
"""

#random_explore & explore_in_cluster（随机探索的比例）
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats.distributions import t as t_distribution
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
pnew_random=np.empty(shape=[0,3])
path=r'D:\results\zxy\new_result\guangzhou\cluster_feature'
filenames=os.listdir(path)
num=0
#S;探索概率；ID
pnew_cluster=np.empty(shape=[0,3])
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
            X=np.array(x)
            hdb = hdbscan.HDBSCAN(min_cluster_size=2,cluster_selection_epsilon=0.01,prediction_data=True).fit(X)
            labels=hdb.labels_
            k=len(x)
            '''
            X=np.array(x)
            hdb = hdbscan.HDBSCAN(min_cluster_size=2,cluster_selection_epsilon=0.01,prediction_data=True).fit(X)
            labels=hdb.labels_
            if labels[-1]==-1:
                s0[k,1]+=1
                s0[k,0]+=1
            else:
                s0[k,1]+=1
            '''
            test_labels, strengths = hdbscan.approximate_predict(hdb, np.array([[data['lon'][i],data['lat'][i]]]))
            if test_labels[0]==-1 and [data['lon'][i],data['lat'][i]] not in x:
                s0[k-1,0]+=1
                s0[k-1,1]+=1
                pnew_random=np.append(pnew_random,[[1,0,k-1]],axis=0)
                
            elif [data['lon'][i],data['lat'][i]] not in x:
                s0[k-1,0]+=1
                s0[k-1,1]+=1
                pnew_random=np.append(pnew_random,[[0,1,k-1]],axis=0)
            else:
                s0[k-1,1]+=1
                
                
        if [data['lon'][i],data['lat'][i]] not in x:
            x+=[[data['lon'][i],data['lat'][i]]]
    k=0
    for j in range(len(s0)):
        if s0[j,1]!=0:
            k+=1
            pnew_cluster=np.append(pnew_cluster,[[k,s0[j,0]/s0[j,1],num]],axis=0)
    num+=1 

pnew_random=np.load(r'D:\results\zxy\new_result\code&preprocessed_data\data\pnew_random.npy')    
def func1(x,rho,b):
    return  x**b
x=[1]
y=[1]
for i in range(3,288):
    x+=[i-2]
    a=pnew_random[pnew_random[:,2]==i]
    y+=[sum(a[:,0])/(sum(a[:,1])+sum(a[:,0]))]

popt,pcov = curve_fit(func1,x[0:40],y[0:40])
print(popt)

fig,ax=plt.subplots(figsize=(7,5),dpi=80)
plt.plot(range(1,99),y[2:100],label='empirical data')
plt.plot(x,[item**(-0.324) for item in x],label=r'$\theta=S^{-0.324}$',linestyle='--',color='red')
plt.xlim(0,40)
plt.ylim(0,1.1)
#plt.yscale('log')
plt.xlabel(r'$S$',size=15)
plt.ylabel(r'$\theta$',size=15)
plt.legend(fontsize=15)  
    
    


rank2_guangzhou=np.load(r'D:\results\zxy\new_result\code&preprocessed_data\data\rank2_guangzhou.npy')    
rank2_houston=np.load(r'D:\results\zxy\new_result\code&preprocessed_data\data\rank2_houston.npy')    
rank2_epr=np.load(r'D:\results\zxy\new_result\code&preprocessed_data\data\rank2_epr.npy')    
def func0(r,k,b):
    return k*r+b
k1=[]
b1=[]
k2=[]
b2=[]
k3=[]
b3=[]
for i in range(1,140):
    x=[]
    y1=[]
    y2=[]
    y3=[]
    for j in range(1,35):
        if rank2_guangzhou[j-1,i-1]!=0 and rank2_houston[j-1,i-1]!=0 and rank2_epr[j-1,i-1]!=0:
            x+=[j]
            y1+=[np.log(rank2_guangzhou[j-1,i-1])]
            y2+=[np.log(rank2_houston[j-1,i-1])]
            y3+=[np.log(rank2_epr[j-1,i-1])]
    popt1,pcov1 = curve_fit(func0,x,y1,maxfev=50000)
    popt2,pcov2 = curve_fit(func0,x,y2,maxfev=50000)
    popt3,pcov3= curve_fit(func0,x,y3,maxfev=50000)
    k1+=[popt1[0]]
    b1+=[popt1[1]]
    k2+=[popt2[0]]
    b2+=[popt2[1]]
    k3+=[popt3[0]]
    b3+=[popt3[1]]
    


plt.plot(range(139),k2,label='houston')
plt.plot(range(139),k1,label='guangzhou')
plt.plot(range(139),k3,label='epr')
plt.legend()

plt.plot(range(139),k1,label='guangzhou')
plt.plot(range(150),[-2.848/(x+1)**0.578 for x in range(150)],label='-2.848(x+1)**(-0.578)')
plt.legend()


plt.plot(range(139),k2,label='houston')
plt.plot(range(150),[-3.572/(x+1)**0.819 for x in range(150)],label='-3.572(x+1)**(-0.819)')
plt.legend()