# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:31:43 2025

@author: ADMIN
"""
# -*- coding: utf-8 -*-
"""
Spatial Frequency Analysis for Fractal Mobility Patterns.
Calculates the total visit frequency within varying radii for different locations.

@author: ADMIN
"""

import os
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


'''
# --- Path Configuration ---
BASE_OUTPUT_PATH = r"D:\results\zxy\new_result\code&preprocessed_data"
DATA_CONFIG = {
    "houston": os.path.join(BASE_OUTPUT_PATH , "houston"),
    "epr": os.path.join(BASE_OUTPUT_PATH , "epr"),
    "cepr": os.path.join(BASE_OUTPUT_PATH , "cepr")
}

# --- Data Preprocessing ---
RADIUS_LIST = [10, 50, 100, 300, 500, 1000, 3000, 5000, 10000, 15000, 20000, 50000, 100000]

def process_spatial_frequency(source_path, output_filename):
    """
    Processes all cluster files in a given directory to calculate 
    spatial frequency distributions across defined radii.
    """
    spatial_freq_result = []
    feature_path = os.path.join(source_path, "cluster_feature")
    cluster_dir = os.path.join(source_path, "cluster")
    
    if not os.path.exists(feature_path):
        print(f"Warning: Path not found {feature_path}")
        return

    filenames = os.listdir(feature_path)

    for filename in tqdm(filenames, desc=f"Processing {output_filename}"):
        subpath = os.path.join(cluster_dir, filename)
        
        # Load and sort data by frequency
        data = pd.read_csv(subpath).sort_values(by=['fre'], ascending=False).reset_index(drop=True)
        
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(data['lon'], data['lat'])]
        gdf = gpd.GeoDataFrame(data, geometry=geometry)
        gdf = gdf.set_crs('WGS84')
        
        # Project to local coordinate system for accurate buffering (meters)
        gdf = gdf.to_crs('EPSG:4520')
        
        points = gdf.geometry
        frequencies = gdf['fre'].values
        point_indices = gdf.index.values 
        
        # Iterate through each location
        for i in range(len(gdf)):
            center_point_geom = points.iloc[i]
            center_freq = frequencies[i]
            center_rank = point_indices[i] + 1  # 1-based rank
            
            # Spatial query for each radius
            for radius in RADIUS_LIST:
                buffer_geom = center_point_geom.buffer(radius)
                is_within = points.within(buffer_geom)
                
                # Sum frequencies within buffer (excluding the center point itself)
                total_freq = frequencies[is_within].sum() - center_freq   
                
                spatial_freq_result.append([
                    center_freq,
                    radius, 
                    total_freq,
                    filenames.index(filename), 
                    center_rank, 
                ])

    # Save results
    result_df = pd.DataFrame(spatial_freq_result, columns=['location_fre', 'r', 'total_fre', 'id', 'location'])
    output_full_path = os.path.join(BASE_OUTPUT_PATH, f"spatial_freq_{output_filename}.txt")
    result_df.to_csv(output_full_path, index=False)
    print(f"Saved: {output_full_path}")

# --- Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    # Process Houston Empirical Data
    process_spatial_frequency(DATA_CONFIG["houston"], "houston")

    # Process EPR Model Results
    process_spatial_frequency(DATA_CONFIG["epr"], "epr")

    # Process Cluster-based EPR (cEPR) Model Results
    process_spatial_frequency(DATA_CONFIG["cepr"], "cepr")

    print("All spatial frequency processing completed.")

'''    


# Streamlined path definitions using os.path.join
BASE_PATH = r"D:\results\zxy\new_result\code&preprocessed_data\data"

data_dict = {
    "houston": pd.read_csv(os.path.join(BASE_PATH, "spatial_freq_houston.txt")),
    "epr": pd.read_csv(os.path.join(BASE_PATH, "spatial_freq_epr.txt")),
    "cepr": pd.read_csv(os.path.join(BASE_PATH, "spatial_freq_cepr.txt"))
}

plt.rcParams['font.sans-serif'] = ['Arial']  
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 2  

def apply_academic_style(ax):
    """Applies clean academic styling rules to the axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=24, direction='out', width=2, length=6)
    ax.tick_params(axis='both', which='minor', direction='out', width=1, length=3)

def plot_scaling_F_vs_f(data):
    """
    Plots Area Frequency (F) as a function of Center Frequency (f) 
    for 7 distinct physical radii. (Lines Only)
    """
    df = data.copy()
    r_list = [100, 300, 500, 1000, 3000, 5000, 10000]
    labels = ['100m', '300m', '500m', '1km', '3km', '5km', '10km']

    num_bins = 15 
    bins = np.logspace(np.log10(1), np.log10(100), num_bins + 1)
    df['f_group'] = pd.cut(df['location_fre'], bins=bins, labels=False, include_lowest=True)

    agg_data = df.groupby(['r', 'f_group']).agg({
        'total_fre': 'mean',
        'location_fre': 'mean' 
    }).reset_index()
    agg_data.columns = ['r', 'f_group', 'F_mean', 'f_mean']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300) 
    colors_f = [cm.viridis(i) for i in np.linspace(0, 0.8, len(labels))]

    for i, r_val in enumerate(r_list):
        subset = agg_data[agg_data['r'] == r_val].sort_values(by='f_mean')
        if len(subset) == 0: 
            continue
            
        ax.plot(subset['f_mean'], subset['F_mean'], 
                linewidth=3, 
                color=colors_f[i],
                label=f'$r = {labels[i]}$')

    apply_academic_style(ax)
    ax.set_ylim(0.1, 1000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Center Frequency ($f$)', fontsize=28)
    ax.set_ylabel('Area Frequency ($F$)', fontsize=28)
    ax.legend(title='Radius ($r$)', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_scaling_F_vs_r(data):
    """
    Plots Area Frequency (F) as a function of Radius (r) 
    grouped by different mathematical frequency bands. (Lines + Scatters)
    """
    df = data[data.r >= 100].copy()
    r_list = [100, 300, 500, 1000, 3000, 5000, 10000]
    bins = [0, 5, 10, 20, 50, 100]
    labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
    
    df['f_group'] = pd.cut(df['location_fre'], bins=bins, labels=labels, include_lowest=True)
    agg_data = df.groupby(['f_group', 'r'], observed=True)['total_fre'].mean().reset_index()
    agg_data.columns = ['f_group', 'r', 'mean']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    colors_f = [cm.plasma(i) for i in np.linspace(0, 0.8, len(labels))]

    for i, group_label in enumerate(labels):
        subset = agg_data[agg_data['f_group'] == group_label].sort_values(by='r')
        if len(subset) == 0: 
            continue
            
        low_b, up_b = bins[i], bins[i+1]
        if i == 0:
            custom_leg = rf'$f < {up_b}$' 
        elif i == len(labels) - 1:
            custom_leg = rf'$f > {low_b}$' 
        else:
            custom_leg = rf'$f \in ({low_b}, {up_b}]$' 

        # Layer 1: Trend Lines
        ax.plot(subset['r'], subset['mean'], linewidth=3, color=colors_f[i], zorder=2)
        # Layer 2: Markers with clean white boundaries
        ax.scatter(subset['r'], subset['mean'], color=colors_f[i], marker='o', s=160, 
                   edgecolors='white', linewidths=0.8, label=custom_leg, zorder=3)

    apply_academic_style(ax)
    ax.set_ylim(0.1, 1100)
    ax.set_xlim(90, 11000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(r_list)
    ax.set_xticklabels([str(x) for x in r_list], rotation=30) 
    ax.set_xlabel('Radius ($r$)', fontsize=28)
    ax.set_ylabel('Area Frequency ($F$)', fontsize=28)
    ax.legend(title='Center Frequency', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=14)
    plt.tight_layout()
    plt.show()


plot_scaling_F_vs_f(data_dict["houston"])    
plot_scaling_F_vs_f(data_dict["cepr"])    
plot_scaling_F_vs_f(data_dict["epr"])    
plot_scaling_F_vs_r(data_dict["houston"])
plot_scaling_F_vs_r(data_dict["cepr"])
plot_scaling_F_vs_r(data_dict["epr"])
    
