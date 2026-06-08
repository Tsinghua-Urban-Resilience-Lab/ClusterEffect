# -*- coding: utf-8 -*-
"""
The cluster effect: How spatial grouping governs human mobility

Core results for Figure 2 in the main text.

@author: Xinyuan, Zhang
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from shapely.geometry import Point, Polygon, shape
import shapely.affinity
import geopandas as gpd
import matplotlib.cm as cm
import statsmodels.api as sm
from scipy.optimize import curve_fit

# --- Function Definitions ---

def ols_regression(x1, y1):
    """Perform OLS regression and return p-value and coefficient."""
    x1 = sm.add_constant(x1)
    mod = sm.OLS(y1, x1)
    res = mod.fit()
    return res.f_pvalue, res.params[1] 

def log_binning(x, y, bins):
    """Apply logarithmic binning for distribution analysis."""
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

def func(t, C, a):
    """Power-law function for curve fitting."""
    return C * (t**a)


def get_linear_stats(x_vals, y_vals, city_name):
    log_x = np.log(list(x_vals)).reshape(-1, 1)
    log_y = np.log(y_vals)
    X = sm.add_constant(log_x)
    res = sm.OLS(log_y, X).fit()
    intercept_val = res.params[0]
    slope_val = res.params[1]
    intercept_se = res.bse[0]
    slope_se = res.bse[1]
    
    print(f"--- {city_name} Zipf's Law Statistics ---")
    print(f"Intercept ln(C): {intercept_val:.4f} ± {intercept_se:.4f}")
    print(f"Slope (Index k): {slope_val:.4f} ± {slope_se:.4f}")
    print(f"R-squared: {res.rsquared:.4f}")
    print(f"P-value: {res.pvalues[1]:.4e}")
    print("-" * 40)



# --- Data Loading and Preprocessing ---

# Setup base path and data directory
base_path = r'D:\results\zxy\new_result\code&preprocessed_data'
data_path = os.path.join(base_path, 'data')

# Load return probability results
return_in_cluster_guangzhou = np.load(os.path.join(data_path, 'return_in_cluster_guangzhou.npy'))
return_in_cluster_houston = np.load(os.path.join(data_path, 'return_in_cluster_houston.npy'))
return_cluster_guangzhou = np.load(os.path.join(data_path, 'return_cluster_guangzhou.npy'))
return_cluster_houston = np.load(os.path.join(data_path, 'return_cluster_houston.npy'))

# Process external pnew data (Guangzhou)
pnew_cluster_guangzhou = np.load(os.path.join(data_path, 'pnew_guangzhou.npy'))
pnew_cluster_guangzhou = pd.DataFrame(pnew_cluster_guangzhou, columns=['s', 'visit', 'visit_explore', 'visit_random', 'locations', 'id'])
pnew_cluster_guangzhou['explore'] = pnew_cluster_guangzhou['visit_explore'] / pnew_cluster_guangzhou['visit']
pnew_cluster_guangzhou['random'] = pnew_cluster_guangzhou['visit_random'] / pnew_cluster_guangzhou['visit']

# Process external pnew data (Houston)
pnew_cluster_houston = np.load(os.path.join(data_path, 'pnew_houston.npy'))
pnew_cluster_houston = pd.DataFrame(pnew_cluster_houston, columns=['s', 'visit', 'visit_explore', 'visit_random', 'locations', 'id'])
pnew_cluster_houston['explore'] = pnew_cluster_houston['visit_explore'] / pnew_cluster_houston['visit']
pnew_cluster_houston['random'] = pnew_cluster_houston['visit_random'] / pnew_cluster_houston['visit']

# Process internal pnew data (Guangzhou)
pnew_in_cluster_guangzhou = np.load(os.path.join(data_path, 'pnew_in_cluster_guangzhou.npy'))
pnew_in_cluster_guangzhou = pd.DataFrame(pnew_in_cluster_guangzhou, columns=['s', 'visit', 'visit_explore', 'cluster', 'cluster_rank', 'cluster_size', 'id'])
pnew_in_cluster_guangzhou = pnew_in_cluster_guangzhou[pnew_in_cluster_guangzhou['cluster_size'] >= 40].reset_index()
pnew_in_cluster_guangzhou['explore'] = pnew_in_cluster_guangzhou['visit_explore'] / pnew_in_cluster_guangzhou['visit']

# Process internal pnew data (Houston)
pnew_in_cluster_houston = np.load(os.path.join(data_path, 'pnew_in_cluster_houston.npy'))
pnew_in_cluster_houston = pd.DataFrame(pnew_in_cluster_houston, columns=['s', 'visit', 'visit_explore', 'cluster', 'cluster_rank', 'cluster_size', 'id'])
pnew_in_cluster_houston = pnew_in_cluster_houston[pnew_in_cluster_houston['cluster_size'] >= 40].reset_index()
pnew_in_cluster_houston['explore'] = pnew_in_cluster_houston['visit_explore'] / pnew_in_cluster_houston['visit']

# Process Zipf's Law data (Houston)
data_houston = np.load(os.path.join(data_path, 'zipf_law_location_houston.npy'))
data_houston = pd.DataFrame(data_houston, columns=['rank', 'fre', 'sumfre', 'locations', 'cluster', 'id'])
data_houston['freq'] = data_houston['fre'] / data_houston['sumfre']
zipf_law_cluster_houston = data_houston[data_houston.locations <= 100].reset_index()

# Process Zipf's Law data (Guangzhou)
data_guangzhou = np.load(os.path.join(data_path, 'zipf_law_location_guangzhou.npy'))
data_guangzhou = pd.DataFrame(data_guangzhou, columns=['rank', 'fre', 'sumfre', 'locations', 'cluster', 'id'])
data_guangzhou['freq'] = data_guangzhou['fre'] / data_guangzhou['sumfre']
zipf_law_cluster_guangzhou = data_guangzhou[data_guangzhou.locations <= 100].reset_index()

# Process internal Zipf's Law data (Houston)
data2_houston = np.load(os.path.join(data_path, 'zipf_law_in_cluster_houston.npy'))
data2_houston = pd.DataFrame(data2_houston, columns=['rank', 'fre', 'sumfre', 'locations', 'cluster', 'id'])
data2_houston['freq'] = data2_houston['fre'] / data2_houston['sumfre']
zipf_law_in_cluster_houston = data2_houston[data2_houston.locations <= 100].reset_index()

# Process internal Zipf's Law data (Guangzhou)
data2_guangzhou = np.load(os.path.join(data_path, 'zipf_law_in_cluster_guangzhou.npy'))
data2_guangzhou = pd.DataFrame(data2_guangzhou, columns=['rank', 'fre', 'sumfre', 'locations', ' ', 'cluster', 'id'])
data2_guangzhou['freq'] = data2_guangzhou['fre'] / data2_guangzhou['sumfre']
zipf_law_in_cluster_guangzhou = data2_guangzhou[data2_guangzhou.locations <= 100].reset_index()

# --- Visualizations ---

# Universal plot settings
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 24

# Figure 2d
x1 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mean1 = [return_in_cluster_guangzhou[0, int(i//0.02)] / return_in_cluster_guangzhou[1, int(i//0.02)] for i in x1]
x2 = [item + 0.01 for item in x1]
mean2 = [return_in_cluster_houston[0, int(i//0.02)] / return_in_cluster_houston[1, int(i//0.02)] for i in x1]

fig = plt.figure(figsize=(9, 6), dpi=300)
plt.scatter(x1, mean1, color='#F76708', marker='^', label='Guangzhou', s=160) 
plt.scatter(x2, mean2, color='#3B3985', marker='^', label='Houston', s=160)
plt.plot([0, 1], [0, 1], linewidth=3, color='gray', linestyle='--', label=r'$fre_l=\pi$')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.xlabel(r'$fre_l$', size=28); plt.ylabel(r'$\pi$', size=28)
plt.show()

# Figure 2a
mean1_2a = [return_cluster_guangzhou[0, int(i//0.02)] / return_cluster_guangzhou[1, int(i//0.02)] for i in x1]
mean2_2a = [return_cluster_houston[0, int(i//0.02)] / return_cluster_houston[1, int(i//0.02)] for i in x1]

fig = plt.figure(figsize=(9, 6), dpi=300)
plt.scatter(x1, mean1_2a, color='#F76708', label='Guangzhou', marker='^', s=160) 
plt.scatter(x2, mean2_2a, color='#3B3985', label='Houston', marker='^', s=160)
plt.plot([0, 1], [0, 1], linewidth=3, color='gray', linestyle='--', label=r'$fre_c=\Pi$')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.xlabel(r'$fre_c$', size=28); plt.ylabel(r'$\Pi$', size=28)
plt.show()

# Figure 2b
x1_range = range(2, 30)
mean1_2b, std1_2b = [], []
for i in x1_range:
    a = pnew_cluster_guangzhou[pnew_cluster_guangzhou['s'] == i]['explore']
    mean1_2b.append(np.mean(a))
    std1_2b.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x, y = log_binning(list(x1_range), mean1_2b, 20)
x, z = log_binning(list(x1_range), std1_2b, 20)
popt, pcov = curve_fit(func, x, y, maxfev=50000)
perr = np.sqrt(np.diag(pcov)) 

# Standard Deviation of the parameters
print("--- Guangzhou Results ---")
print(f"C: {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"a: {popt[1]:.4f} ± {perr[1]:.4f}")


mean2_2b, std2_2b = [], []
for i in x1_range:
    a = pnew_cluster_houston[pnew_cluster_houston['s'] == i]['explore']
    mean2_2b.append(np.mean(a))
    std2_2b.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x2, y2 = log_binning(list(x1_range), mean2_2b, 25)
x2, z2 = log_binning(list(x1_range), std2_2b, 25)
popt2, pcov2 = curve_fit(func, x2, y2, maxfev=50000)
perr2 = np.sqrt(np.diag(pcov2)) 
print("\n--- Houston Results ---")
print(f"C: {popt2[0]:.4f} ± {perr2[0]:.4f}")
print(f"a: {popt2[1]:.4f} ± {perr2[1]:.4f}")


fig = plt.figure(figsize=(9, 6), dpi=300)
plt.plot(range(1, 31), [(popt[0])*(i**(popt[1])) for i in range(1, 31)], label=r'$0.57S^{-0.28}$', c='#F76708')     
plt.scatter(x, y, color='#F76708', marker='^', s=160)
plt.errorbar(x=x, y=y, yerr=z, fmt='.k', elinewidth=2, markersize=0, ecolor='#F76708')
plt.plot(range(1, 31), [(popt2[0])*(i**(popt2[1])) for i in range(1, 31)], label=r'$0.62S^{-0.26}$', c='#3B3985')  
plt.scatter([i+0.05*i for i in x2], y2, color='#3B3985', marker='^', s=160)
plt.errorbar(x=[i+0.05*i for i in x2], y=y2, yerr=z2, fmt='.k', elinewidth=2, markersize=0, ecolor='#3B3985')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.ylim(0.1, 1.5); plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$S$', size=28); plt.ylabel(r'$P_{new}^{external}$', size=28, labelpad=-8)
plt.show()

# Figure 2e
x1_range_2e = range(1, 31)
mean1_2e, std1_2e = [], []
for i in x1_range_2e:
    a = pnew_in_cluster_guangzhou[pnew_in_cluster_guangzhou['s'] == i]['explore']
    mean1_2e.append(np.mean(a))
    std1_2e.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x, y = log_binning(list(x1_range_2e), mean1_2e, 25)
x, z = log_binning(list(x1_range_2e), std1_2e, 25)
popt, pcov = curve_fit(func, x, y, maxfev=50000)
perr = np.sqrt(np.diag(pcov)) 

# Standard Deviation of the parameters
print("--- Guangzhou Results ---")
print(f"C: {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"a: {popt[1]:.4f} ± {perr[1]:.4f}")


mean2_2e, std2_2e = [], []
for i in x1_range_2e:
    a = pnew_in_cluster_houston[pnew_in_cluster_houston['s'] == i]['explore']
    mean2_2e.append(np.mean(a))
    std2_2e.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x2, y2 = log_binning(list(x1_range_2e), mean2_2e, 25)
x2, z2 = log_binning(list(x1_range_2e), std2_2e, 25)
popt2, pcov = curve_fit(func, x2, y2, maxfev=50000)
perr2 = np.sqrt(np.diag(pcov2)) 
print("\n--- Houston Results ---")
print(f"C: {popt2[0]:.4f} ± {perr2[0]:.4f}")
print(f"a: {popt2[1]:.4f} ± {perr2[1]:.4f}")


fig = plt.figure(figsize=(9, 6), dpi=300)
plt.plot(range(1, 31), [(popt[0])*(i**(popt[1])) for i in range(1, 31)], label=r'$0.75s^{-0.32}$', c='#F76708')     
plt.scatter(x, y, color='#F76708', marker='^', s=160)
plt.errorbar(x=x, y=y, yerr=z, fmt='.k', elinewidth=2, markersize=0, ecolor='#F76708')
plt.plot(range(1, 31), [(popt2[0])*(i**(popt2[1])) for i in range(1, 31)], label=r'$0.68s^{-0.29}$', c='#3B3985')  
plt.scatter([i+0.05*i for i in x2], y2, color='#3B3985', marker='^', s=160)
plt.errorbar(x=[i+0.05*i for i in x2], y=y2, yerr=z2, fmt='.k', elinewidth=2, markersize=0, ecolor='#3B3985')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.ylim(0.1, 1.5); plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$S_c$', size=28); plt.ylabel(r'$P_{new}^{internal}$', size=28, labelpad=-8)
plt.show()

# Figure 2c
x1_range_2c = range(1, 31)
mean1_2c, std1_2c = [], []
for i in x1_range_2c:
    a = zipf_law_cluster_houston[zipf_law_cluster_houston['rank'] == i]['freq']
    mean1_2c.append(np.mean(a))
    std1_2c.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x, y = log_binning(list(x1_range_2c), mean1_2c, 25)
x, z = log_binning(list(x1_range_2c), std1_2c, 25)

mean2_2c, std2_2c = [], []
for i in x1_range_2c:
    a = zipf_law_cluster_guangzhou[zipf_law_cluster_guangzhou['rank'] == i]['freq']
    mean2_2c.append(np.mean(a))
    std2_2c.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x2, y2 = log_binning(list(x1_range_2c), mean2_2c, 25)
x2, z2 = log_binning(list(x1_range_2c), std2_2c, 25)

model = LinearRegression()
model.fit(np.log(list(x1_range_2c)).reshape(-1, 1), np.log(mean1_2c))
k1, a1 = -model.coef_[0], model.intercept_

model.fit(np.log(list(x1_range_2c)).reshape(-1, 1), np.log(mean2_2c))
k2, a2 = -model.coef_[0], model.intercept_

get_linear_stats(x1_range_2c, mean1_2c, "Houston (Fig 2c)")
get_linear_stats(x1_range_2c, mean2_2c, "Guangzhou (Fig 2c)")

fig = plt.figure(figsize=(9, 6), dpi=300)
plt.plot(range(1, 31), [(np.exp(a1))*(i**(-k1)) for i in range(1, 31)], label=r'$k_{l}^{-1.60}$', c='#F76708')     
plt.scatter(x, y, color='#F76708', marker='^', s=160)
plt.errorbar(x=x, y=y, yerr=z, fmt='.k', elinewidth=2, markersize=0, ecolor='#F76708')
plt.plot(range(1, 31), [(np.exp(a2))*(i**(-k2)) for i in range(1, 31)], label=r'$k_{l}^{-1.58}$', c='#3B3985')  
plt.scatter([i+0.05*i for i in x2], y2, color='#3B3985', marker='^', s=160)
plt.errorbar(x=[i+0.05*i for i in x2], y=y2, yerr=z2, fmt='.k', elinewidth=2, markersize=0, ecolor='#3B3985')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.xlim(0.8, 35); plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$k_l$', size=28); plt.ylabel(r'$fre_{k_{l}}$', size=28, labelpad=-8)
plt.show()

# Figure 2f
x1_range_2f = range(1, 31)
mean1_2f, std1_2f = [], []
for i in x1_range_2f:
    a = zipf_law_in_cluster_houston[zipf_law_in_cluster_houston['rank'] == i]['freq']
    mean1_2f.append(np.mean(a))
    std1_2f.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x, y = log_binning(list(x1_range_2f), mean1_2f, 25)
x, z = log_binning(list(x1_range_2f), std1_2f, 25)

mean2_2f, std2_2f = [], []
for i in x1_range_2f:
    a = zipf_law_in_cluster_guangzhou[zipf_law_in_cluster_guangzhou['rank'] == i]['freq']
    mean2_2f.append(np.mean(a))
    std2_2f.append(2 * 1.96 * np.std(a, ddof=1) / math.sqrt(len(a)))

x2, y2 = log_binning(list(x1_range_2f), mean2_2f, 25)
x2, z2 = log_binning(list(x1_range_2f), std2_2f, 25)

model.fit(np.log(list(x1_range_2f)).reshape(-1, 1), np.log(mean1_2f))
k1_f, a1_f = -model.coef_[0], model.intercept_

model.fit(np.log(list(x1_range_2f)).reshape(-1, 1), np.log(mean2_2f))
k2_f, a2_f = -model.coef_[0], model.intercept_

get_linear_stats(x1_range_2f, mean1_2f, "Houston (Fig 2f)")
get_linear_stats(x1_range_2f, mean2_2f, "Guangzhou (Fig 2f)")

fig = plt.figure(figsize=(9, 6), dpi=300)
plt.plot(range(1, 31), [(np.exp(a1_f))*(i**(-k1_f)) for i in range(1, 31)], label=r'$k_{l\ in\ c}^{-1.66}$', c='#F76708')     
plt.scatter(x, y, color='#F76708', marker='^', s=160)
plt.errorbar(x=x, y=y, yerr=z, fmt='.k', elinewidth=2, markersize=0, ecolor='#F76708')
plt.plot(range(1, 31), [(np.exp(a2_f))*(i**(-k2_f)) for i in range(1, 31)], label=r'$k_{l\ in\ c}^{-1.57}$', c='#3B3985')  
plt.scatter([i+0.05*i for i in x2], y2, color='#3B3985', marker='^', s=160)
plt.errorbar(x=[i+0.05*i for i in x2], y=y2, yerr=z2, fmt='.k', elinewidth=2, markersize=0, ecolor='#3B3985')
plt.yticks(size=24); plt.xticks(size=24)
plt.legend(frameon=False, fontsize=24)
plt.xlim(0.8, 35); plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$k_{l\ in\ c}$', size=28); plt.ylabel(r'$fre_{k_{l\ in\ c}}$', size=28, labelpad=-8)
plt.show()