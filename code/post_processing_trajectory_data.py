# -*- coding: utf-8 -*-
"""
Post-processing of simulated trajectory data
Step 1: Pattern extraction and frequency calculation.
Step 2: HDBSCAN clustering and feature extraction.
Step 3: Cluster-location information mapping.

@author: Xinyuan, Zhang
"""

import os
import math
import numpy as np
import pandas as pd
import hdbscan
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm

# --- 1. Path Configuration ---
BASE_PATH = r'D:\results\zxy\new_result\epr'
RAW_P_DIR = os.path.join(BASE_PATH, 'p')
CALC_DIR = os.path.join(BASE_PATH, 'calculate')
PATTERN_DIR = os.path.join(BASE_PATH, 'pattern')
FEATURE_DIR = os.path.join(BASE_PATH, 'cluster_feature')
CLUSTER_DIR = os.path.join(BASE_PATH, 'cluster')

# Ensure all target directories exist
for d in [CALC_DIR, PATTERN_DIR, FEATURE_DIR, CLUSTER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 2. Geometric Calculation Functions ---
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points in meters."""
    # Using 6367000m to maintain consistency with original code
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 6367000 * 2 * asin(sqrt(a))

def calculate_metrics(df):
    """Calculate Radius of Gyration (Rg), center of mass, and max radius."""
    if len(df) == 0: return 0, 0, 0, 0
    clon, clat = df['lon'].mean(), df['lat'].mean()
    # Efficient calculation of distances using apply
    dists = df.apply(lambda r: haversine(r.lon, r.lat, clon, clat), axis=1)
    rg = math.sqrt((dists**2).sum() / len(df))
    return rg, clon, clat, dists.max()

# --- STEP 1: Trajectory Formatting and Pattern Extraction ---
print("Step 1: Processing Patterns...")
filenames = [f for f in os.listdir(RAW_P_DIR) if f.endswith('.txt')]

for filename in tqdm(filenames, desc="Pattern Extraction", unit="file"):
    sub_path = os.path.join(RAW_P_DIR, filename)
    
    # Read raw data with explicit float dtypes to prevent LossySetitemError
    data = pd.read_csv(sub_path, header=None, usecols=[0, 1, 2, 3], 
                       names=['arrive', 'leave', 'lon', 'lat'],
                       dtype={'arrive': float, 'leave': float, 'lon': float, 'lat': float})
    
    # Time unit conversion and feature engineering
    data['arrive_time'] = data['arrive'] * 3600
    data['leave_time'] = data['leave'] * 3600
    data['duration'] = data['leave_time'] - data['arrive_time']
    
    for t in ['arrive', 'leave']:
        data[f'{t}_day'] = (data[t] // 24).astype(int)
        data[f'{t}_hour'] = (data[t] % 24).astype(int)

    # Use coordinates to define unique locations (5 decimal places precision)
    data['loc_key'] = data.apply(lambda r: f"[{r.lon:.5f},{r.lat:.5f}]", axis=1)
    unique_locs = data['loc_key'].unique().tolist()
    data['location'] = data['loc_key'].map(unique_locs.index)

    # 1.1 Generate location-level summary for 'calculate' folder
    calc_df = data.groupby('location').agg({
        'lon': 'first', 'lat': 'first', 'location': 'count', 'duration': 'sum'
    }).rename(columns={'location': 'fre', 'duration': 'time'}).reset_index()
    
    # Sort and save based on frequency
    calc_df = calc_df[['location', 'lon', 'lat', 'fre', 'time']].sort_values('fre')
    calc_df.to_csv(os.path.join(CALC_DIR, filename.replace('.txt', '.csv')), index=False)

    # 1.2 Save full trajectory with location indices to 'pattern' folder
    pattern_cols = ['arrive_time', 'arrive_day', 'arrive_hour', 'leave_time', 
                    'leave_day', 'leave_hour', 'lon', 'lat', 'location']
    data[pattern_cols].to_csv(os.path.join(PATTERN_DIR, filename.replace('.txt', '.csv')), index=False)

# --- STEP 2: Spatial Clustering and Feature Extraction ---
print("\nStep 2: Processing Clustering...")
pattern_files = os.listdir(PATTERN_DIR)

for filename in tqdm(pattern_files, desc="Clustering & Features", unit="file"):
    p_path = os.path.join(PATTERN_DIR, filename)
    data = pd.read_csv(p_path)
    
    # Clustering on unique points to improve computational efficiency
    unique_pts = data[['lon', 'lat', 'location']].drop_duplicates()
    
    if len(unique_pts) > 3:
        # Execute HDBSCAN (eps=0.01 is approx 1km in degree distance)
        hdb = hdbscan.HDBSCAN(min_cluster_size=3, cluster_selection_epsilon=0.01).fit(unique_pts[['lon', 'lat']])
        
        # Map cluster labels back to the full trajectory
        loc_map = dict(zip(unique_pts['location'], hdb.labels_))
        data['cluster'] = data['location'].map(loc_map)
        data.to_csv(p_path, index=False) # Update the pattern file
        
        # Extract cluster-level characteristics
        cluster_list = []
        for cid in [c for c in data['cluster'].unique() if c != -1]: # Skip noise (-1)
            seq = data[data['cluster'] == cid].reset_index(drop=True)
            rg, clon, clat, rmax = calculate_metrics(seq)
            
            cluster_list.append({
                'cluster': cid, 'center_lon': clon, 'center_lat': clat,
                'fre': len(seq), 'time': (seq['leave_time'] - seq['arrive_time']).sum(),
                'rg': rg, 'max_radius': rmax, 'locations': seq['location'].nunique(),
                'starttime': seq['arrive_time'].iloc[0], 'endtime': seq['leave_time'].iloc[-1]
            })
        
        if cluster_list:
            feat_df = pd.DataFrame(cluster_list).sort_values('fre')
            feat_df.to_csv(os.path.join(FEATURE_DIR, filename), index=False)

# --- STEP 3: Final Mapping and Updates ---
print("\nStep 3: Processing Final Integration...")
feature_files = os.listdir(FEATURE_DIR)

for filename in tqdm(feature_files, desc="Final Integration", unit="file"):
    p_df = pd.read_csv(os.path.join(PATTERN_DIR, filename))
    f_df = pd.read_csv(os.path.join(FEATURE_DIR, filename))
    
    # 3.1 Build final location table with cluster center info
    c_df = p_df.groupby('location').agg({
        'lon': 'first', 'lat': 'first', 'cluster': 'first'
    }).reset_index()
    
    # Calculate stats per location
    c_df['fre'] = p_df.groupby('location').size().values
    c_df['t'] = p_df.groupby('location').apply(lambda g: (g['leave_time'] - g['arrive_time']).sum()).values
    
    # Map center coordinates from the feature file
    c_df['center_lon'] = c_df['cluster'].map(dict(zip(f_df['cluster'], f_df['center_lon']))).fillna(0)
    c_df['center_lat'] = c_df['cluster'].map(dict(zip(f_df['cluster'], f_df['center_lat']))).fillna(0)
    
    # 3.2 Correct location count in the feature file
    f_df['locations'] = f_df['cluster'].map(p_df.groupby('cluster')['location'].nunique())
    
    # Save final synchronized results
    f_df.to_csv(os.path.join(FEATURE_DIR, filename), index=False)
    c_df.to_csv(os.path.join(CLUSTER_DIR, filename), index=False)

print("\nAll processing steps completed successfully.")

