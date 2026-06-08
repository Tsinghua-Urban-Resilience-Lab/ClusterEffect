# -*- coding: utf-8 -*-
"""
The cluster effect: How spatial grouping governs human mobility
Core results for Figure 2 and extended analysis.

@author: Xinyuan, Zhang
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm

# --- Function Definitions ---

def ols_regression(x1, y1):
    """Perform OLS regression and return p-value and coefficient."""
    x1 = sm.add_constant(x1)
    mod = sm.OLS(y1, x1)
    res = mod.fit()
    return res.f_pvalue, res.params[1]

def log_binning(x, y, bins):
    """Perform logarithmic binning for spatial data analysis."""
    interval = list(np.logspace(np.log10(min(x)), np.log10(len(x)+1+10), bins, base=10))
    x_list = [min(x)]
    y_list = [y[0]]
    for i in range(1, len(interval)-1):
        sequence = []
        for k in range(len(x)):
            if x[k] <= (interval[i+1]+interval[i])/2 and x[k] > (interval[i]+interval[i-1])/2:
                sequence += [y[k]]
        if len(sequence) != 0:
            x_list += [interval[i]]
            y_list += [np.mean(sequence)]
    return x_list, y_list

def func2(t, C, a):
    """Power-law function for curve fitting (C and a as parameters)."""
    return C * (t**a)

def plot_scene_results(rank1_data, rank2_data, scene_name):
    """
    Generates 4 plots for a specific scene (e.g., Guangzhou or Houston).
    1 & 2: Heatmaps for rank1 and rank2.
    3 & 4: Line plots (Log-scale) for rank1 and rank2 with different rank slices.
    """
    plt.rcParams['font.sans-serif'] = 'Arial'
    
    # --- 1. Heatmap for Rank 1 ---
    fig1 = plt.figure(figsize=(6, 4), dpi=600)
    plt.title(scene_name, size=14, x=0.1, y=1.01)
    sns.heatmap(rank1_data, norm=LogNorm(vmax=0.2), cmap="YlGnBu", 
                cbar_kws={'ticks': MaxNLocator(2), 'format': '%.1f'})
    plt.xlabel(r'$k_l$', size=13)
    plt.ylabel(r'$k_c$', size=13)
    plt.xlim(0, 145); plt.ylim(35, 0)
    # Adding probability vertical label
    plt.text(x=172, y=10, s='Probability', rotation=90, ha='left', va='center',
             fontdict=dict(fontsize=12, family='monospace', weight='light'))
    plt.show()

    # --- 2. Heatmap for Rank 2 ---
    fig2 = plt.figure(figsize=(6, 4), dpi=600)
    plt.title(scene_name, size=14, x=0.1, y=1.01)
    sns.heatmap(rank2_data, norm=LogNorm(vmax=0.2), cmap="YlGnBu", 
                cbar_kws={'ticks': MaxNLocator(2), 'format': '%.1f'})
    plt.xlabel(r'$k_l$', size=13)
    plt.ylabel(r'$k_c$', size=13)
    plt.xlim(0, 145); plt.ylim(35, 0)
    plt.text(x=172, y=10, s='Probability', rotation=90, ha='left', va='center',
             fontdict=dict(fontsize=12, family='monospace', weight='light'))
    plt.show()

    # --- 3. Line Plot for Location Rank (slices of Rank 1) ---
    fig3 = plt.figure(figsize=(9, 3), dpi=1000)
    j = 0
    # Original rank slices: [1, 5, 10, 15, 20, 25, 30]
    for i in [0, 4, 9, 14, 19, 24, 29]:
        x_line, y_line = [], []
        for k in range(1, 146):
            if rank1_data[i, k-1] != 0:
                x_line.append(k)
                y_line.append(rank1_data[i, k-1])
        plt.plot(x_line, y_line, color='#845EC2', alpha=1-0.15*j, linewidth=4-0.4*j, label='rank'+str(i+1))
        j += 1
    plt.yscale('log')
    plt.xlabel('Location rank', size=28); plt.ylabel('Probability', size=28)
    plt.yticks(size=24); plt.xticks(size=24)
    plt.show()

    # --- 4. Line Plot for Cluster Rank (slices of Rank 2) ---
    fig4 = plt.figure(figsize=(9, 3), dpi=1000)
    j = 0
    # Original rank slices: [1, 5, 10, 20, 40, 60]
    for i in [0, 4, 9, 19, 39, 59]:
        x_line, y_line = [], []
        for k in range(1, 36):
            if rank2_data[k-1, i] != 0:
                x_line.append(k)
                y_line.append(rank2_data[k-1, i])
        plt.plot(x_line, y_line, color='#845EC2', alpha=1-0.15*j, linewidth=4-0.4*j, label='rank'+str(i+1))
        j += 1
    plt.yscale('log')
    plt.xlabel('Cluster rank', size=28); plt.ylabel('Probability', size=28)
    plt.yticks(size=24); plt.xticks(size=24)
    plt.show()

# --- Data Loading ---

base_path = r'D:\results\zxy\new_result\code&preprocessed_data'
data_path = os.path.join(base_path, 'data')

# Load Guangzhou datasets
rank1_guangzhou = np.load(os.path.join(data_path, 'rank1_guangzhou.npy'))
rank2_guangzhou = np.load(os.path.join(data_path, 'rank2_guangzhou.npy'))

# Load Houston datasets
rank1_houston = np.load(os.path.join(data_path, 'rank1_houston.npy'))
rank2_houston = np.load(os.path.join(data_path, 'rank2_houston.npy'))

# Load EPR simulation results
rank1_epr = np.load(os.path.join(data_path, 'rank1_epr.npy'))
rank2_epr = np.load(os.path.join(data_path, 'rank2_epr.npy'))

# Load Guangzhou datasets
rank1_cepr = np.load(os.path.join(data_path, 'rank1_cepr_guangzhou.npy'))
rank2_cepr = np.load(os.path.join(data_path, 'rank2_cepr_guangzhou.npy'))

# Load Guangzhou datasets
rank1_cepr2 = np.load(os.path.join(data_path, 'rank1_cepr_houston.npy'))
rank2_cepr2 = np.load(os.path.join(data_path, 'rank2_cepr_houston.npy'))



# --- Execute Visualization ---

# Generate 4 plots for Guangzhou
plot_scene_results(rank1_guangzhou, rank2_guangzhou, "Guangzhou")

# Generate 4 plots for Houston
plot_scene_results(rank1_houston, rank2_houston, "Houston")


plot_scene_results(rank1_epr, rank2_epr, "EPR model")


plot_scene_results(rank1_cepr, rank2_cepr, "Cluster-based model(Guangzhou)")

plot_scene_results(rank1_cepr2, rank2_cepr2, "Cluster-based model(Houston)")


# --- Calculate Correlation Coefficient ---

'''
x=range(1,145)
y_real=[]
y_predict1=[]
y_predict2=[]
for k in x:
    y_real_=0
    y_predict1_=0
    y_predict2_=0
    for j in range(20):
        y_real_+=(j+1)*rank2_houston[j,k-1]
        y_predict1_+=(j+1)*rank2_epr[j,k-1]
        y_predict2_+=(j+1)*rank2_cepr2[j,k-1]
    y_real+=[y_real_]
    y_predict1+=[y_predict1_]
    y_predict2+=[y_predict2_]

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] =12
fig=plt.figure(figsize=(6,4),dpi=600)
plt.scatter(y_real,y_predict1,label='EPR  model',s=5)
plt.scatter(y_real,y_predict2,label='CEPR model',s=5)
plt.plot([min(y_real),max(y_real)],[min(y_real),max(y_real)],color='gray',linestyle='--',linewidth=2)
plt.xlabel('Empirical Value')
plt.ylabel('Predicted Value')
plt.legend()

from scipy import stats

pearson_r, p_pearson = stats.pearsonr(y_real, y_predict2)
df = len(y_real) - 2
pearson_r2, p_pearson2 = stats.pearsonr(y_real, y_predict1)

print(f"EPR Pearson r: {pearson_r2:.3f}, p-value: {p_pearson2}")
print(f"Cluster-based model Pearson r: {pearson_r:.3f}, p-value: {p_pearson}")
'''