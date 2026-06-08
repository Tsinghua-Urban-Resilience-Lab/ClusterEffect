# -*- coding: utf-8 -*-
"""
The scale-invariance of inter- and intra- mobility
@author: Xinyuan, Zhang
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# =============================================================================
# 0. GLOBAL CONFIGURATION & COLORS
# =============================================================================
DATA_ROOT = r"D:\results\zxy\new_result\code&preprocessed_data\data"
C_EXT = '#D62728'  # Vivid Orange (External/Inter)
C_INT = '#2CA02C'  # Deep Purple (Internal/Intra)

# =============================================================================
# 1. CORE FUNCTIONS
# =============================================================================

def log_binning(x, y, bins=25):
    """Performs logarithmic binning for smoothing power-law data."""
    x, y = np.array(x), np.array(y)
    # Adjustment for Zipf's law index k=1
    interval = np.logspace(np.log10(x.min()), np.log10(x.max() + 11), bins)
    x_list, y_list = [x.min()], [y[0]]
    for i in range(1, len(interval) - 1):
        mask = (x <= (interval[i+1] + interval[i]) / 2) & (x > (interval[i] + interval[i-1]) / 2)
        if np.any(mask):
            x_list.append(interval[i])
            y_list.append(np.mean(y[mask]))
    return np.array(x_list), np.array(y_list)

def power_law_func(t, C, a):
    """Fitting function for P_new: C * t^a"""
    return C * (t**a)

def load_and_preprocess_pnew(filepath, is_internal=False):
    """Standardized data loading for P_new."""
    data = np.load(filepath)
    if is_internal:
        cols = ['s', 'visit', 'visit_explore', 'cluster', 'cluster_rank', 'cluster_size', 'id']
        df = pd.DataFrame(data[:, :7], columns=cols)
        df = df[df['cluster_size'] >= 40].reset_index()
    else:
        cols = ['s', 'visit', 'visit_explore', 'visit_random', 'locations', 'id']
        df = pd.DataFrame(data, columns=cols)
    df['explore'] = df['visit_explore'] / df['visit']
    return df

def get_binned_stats_pnew(df, x_range, bins=25):
    """Calculates means, confidence intervals, and applies log binning for P_new."""
    means, cis = [], []
    for i in x_range:
        subset = df[df['s'] == i]['explore']
        means.append(np.mean(subset))
        cis.append(2 * 1.96 * np.std(subset, ddof=1) / math.sqrt(len(subset)))
    
    bx, by = log_binning(list(x_range), means, bins)
    _, bz = log_binning(list(x_range), cis, bins)
    return bx, by, bz

def load_zipf(filename):
    """Standardized data loading for Zipf's Law."""
    data = np.load(os.path.join(DATA_ROOT, filename))
    # Handle different column structures
    if data.shape[1] == 6:
        cols = ['rank','fre','sumfre','locations','cluster','id']
    else:
        cols = ['rank','fre','sumfre','locations','cluster_rank','cluster','id']
    df = pd.DataFrame(data, columns=cols)
    df['freq'] = df['fre'] / df['sumfre']
    return df[df.locations <= 100].reset_index()

def get_zipf_stats(df, x_range):
    """Calculates mean frequency and 95% CI per rank."""
    means, cis = [], []
    for i in x_range:
        subset = df[df['rank'] == i]['freq']
        means.append(np.mean(subset))
        cis.append(2 * 1.96 * np.std(subset, ddof=1) / math.sqrt(len(subset)))
    return np.array(means), np.array(cis)

# =============================================================================
# 2. ANALYSIS PART I: P_NEW MODEL COMPARISON
# =============================================================================
print("Processing P_new plots...")

# Data Ingestion
hou_ext = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_houston.npy"))
hou_int = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_in_cluster_houston.npy"), True)
epr_ext = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_epr.npy"))
epr_int = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_in_cluster_epr.npy"), True)
cepr_ext = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_cepr.npy"))
cepr_int = load_and_preprocess_pnew(os.path.join(DATA_ROOT, "pnew_in_cluster_cepr.npy"), True)

models_pnew = [
    (hou_ext, hou_int, 'Empirical data', 1.5, r'$P_{new}^{external} \sim 0.62s^{-0.26}$', r'$P_{new}^{internal} \sim 0.68s^{-0.29}$'),
    (epr_ext, epr_int, 'EPR model', 1.7, r'$P_{new}^{external} \sim 0.79s^{-0.15}$', r'$P_{new}^{internal} \sim 0.63s^{-0.37}$'),
    (cepr_ext, cepr_int, 'Cluster-based model', 3.7, r'$P_{new}^{external} \sim 0.78s^{-0.15}$', r'$P_{new}^{internal} \sim 0.73s^{-0.17}$')
]

for m_ext, m_int, title, ylim_max, leg_ext, leg_int in models_pnew:
    x_r = range(1, 31)
    bx1, by1, bz1 = get_binned_stats_pnew(m_ext, x_r, 25)
    bx2, by2, bz2 = get_binned_stats_pnew(m_int, x_r, 25)
    
    popt1, _ = curve_fit(power_law_func, bx1, by1)
    popt2, _ = curve_fit(power_law_func, bx2, by2)
    
    fig = plt.figure(figsize=(8, 6), dpi=1000)
    
    # External - Vivid Orange
    plt.plot(range(1, 31), [power_law_func(i, *popt1) for i in range(1, 31)], label=leg_ext, c=C_EXT, lw=2.5)
    plt.scatter(bx1, by1, color=C_EXT, marker='^', s=160, edgecolors='k', linewidths=0.5, zorder=3)
    plt.errorbar(bx1, by1, yerr=bz1, fmt='none', elinewidth=2, markersize=0, ecolor=C_EXT, alpha=0.7)
    
    # Internal - Deep Purple (shifted 5%)
    plt.plot(range(1, 31), [power_law_func(i, *popt2) for i in range(1, 31)], label=leg_int, c=C_INT, lw=2.5, linestyle='--')
    plt.scatter([i * 1.05 for i in bx2], by2, color=C_INT, marker='o', s=160, edgecolors='k', linewidths=0.5, zorder=3)
    plt.errorbar([i * 1.05 for i in bx2], by2, yerr=bz2, fmt='none', elinewidth=2, markersize=0, ecolor=C_INT, alpha=0.7)
    
    plt.xscale('log'); plt.yscale('log')
    plt.xticks(size=24); plt.yticks(size=24)
    plt.xlabel(r'$S$', size=28); plt.ylabel(r'$P_{new}$', size=28, labelpad=-8)
    plt.ylim(0.1, ylim_max)
    plt.legend(frameon=False, fontsize=24, loc='upper right')
    plt.text(x=1, y=ylim_max, s=title, ha='left', va='bottom', fontdict={'fontsize': 24, 'weight': 'light'})
    plt.show()

# =============================================================================
# 3. ANALYSIS PART II: ZIPF'S LAW COMPARISON
# =============================================================================
print("Processing Zipf's Law plots...")

def plot_zipf_comparison_refined(df_ext, df_int, title, exponent_labels):
    x_range = np.arange(1, 31)
    m1, e1 = get_zipf_stats(df_ext, x_range)
    m2, e2 = get_zipf_stats(df_int, x_range)
    bx1, by1 = log_binning(x_range, m1)
    _, bz1 = log_binning(x_range, e1)
    bx2, by2 = log_binning(x_range, m2)
    _, bz2 = log_binning(x_range, e2)

    def get_fit(x, y):
        model = LinearRegression().fit(np.log(x).reshape(-1, 1), np.log(y))
        return -model.coef_[0], np.exp(model.intercept_)

    k1, a1 = get_fit(x_range, m1)
    k2, a2 = get_fit(x_range, m2)

    fig = plt.figure(figsize=(8, 6), dpi=1000)
    
    # Location (External) - Vivid Orange
    plt.plot(x_range, a1 * (x_range**-k1), label=exponent_labels[0], c=C_EXT, lw=2.5)
    plt.scatter(bx1, by1, color=C_EXT, marker='^', s=160, edgecolors='k', linewidths=0.5, zorder=3)
    plt.errorbar(bx1, by1, yerr=bz1, fmt='none', elinewidth=2, markersize=0, ecolor=C_EXT, alpha=0.7)
    
    # In-cluster (Internal) - Deep Purple (shifted 5%)
    plt.plot(x_range, a2 * (x_range**-k2), label=exponent_labels[1], c=C_INT, lw=2.5, linestyle='--')
    plt.scatter(bx2 * 1.05, by2, color=C_INT, marker='o', s=160, edgecolors='k', linewidths=0.5, zorder=3)
    plt.errorbar(bx2 * 1.05, by2, yerr=bz2, fmt='none', elinewidth=2, markersize=0, ecolor=C_INT, alpha=0.7)

    plt.xscale('log'); plt.yscale('log')
    plt.xticks(size=24); plt.yticks(size=24)
    plt.ylim(0.0001, 1)
    plt.xlabel(r'$k$', size=28, labelpad=-4)
    plt.ylabel(r'$fre_{k}$', size=28, labelpad=-4)
    plt.legend(frameon=False, fontsize=24, loc='upper right')
    plt.text(0.9, 1, title, ha='left', va='bottom', fontdict={'fontsize': 24, 'weight': 'light'})
    plt.show()

# Execution for Zipf's Law
df_hou_ext_z = load_zipf("zipf_law_location_houston.npy")
df_hou_int_z = load_zipf("zipf_law_in_cluster_houston.npy")
plot_zipf_comparison_refined(df_hou_ext_z, df_hou_int_z, 'Empirical data', [r'$k_{l}^{-1.58}$', r'$k_{l\,in\,c}^{-1.57}$'])

df_epr_ext_z = load_zipf("zipf_law_location_epr.npy")
df_epr_int_z = load_zipf("zipf_law_in_cluster_epr.npy")
plot_zipf_comparison_refined(df_epr_ext_z, df_epr_int_z, 'EPR model', [r'$k_{l}^{-1.02}$', r'$k_{l\,in\,c}^{-1.57}$'])

df_cepr_ext_z = load_zipf("zipf_law_location_cepr2.npy")
df_cepr_int_z = load_zipf("zipf_law_in_cluster_cepr2.npy")
plot_zipf_comparison_refined(df_cepr_ext_z, df_cepr_int_z, 'Cluster-based model', [r'$k_{l}^{-1.07}$', r'$k_{l\,in\,c}^{-1.18}$'])

print("All simulations and visualizations complete.")