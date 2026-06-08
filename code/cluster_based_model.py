# -*- coding: utf-8 -*-
"""
The cluster effect: How spatial grouping governs human mobility
Individual Mobility Simulation using the proposed cluster-based model

@author: Xinyuan, Zhang
"""
import os
import math
import random
import time
import numpy as np
import pandas as pd
import hdbscan
from math import radians, cos, sin, asin, sqrt

#Parameter Setting
#For Guangzhou: alpha,beta,rho,gamma,tau,omega,epsilon,lon_min,lon_max,lat_min,lat_max=0.615,0.618,0.571,0.283,0.595,-2.848,0.578,112.95,114.05,22.43,23.93
#For Houston:alpha,beta,rho,gamma,lon_min,lon_max,lat_min,lat_max=0.415,0.800,0.618,0.264,0.606,-3.572,0.819,-96.13,-94.69,29.37,30.29

# --- Model Parameters ---
alpha = 0.615   
beta = 0.618    
rho = 0.571  
gamma = 0.283
tau=0.595
omega=-0.2848
epsilon=0.578

# Spatial Boundaries (Guangzhou)
lon_min, lon_max = 112.95, 114.05
lat_min, lat_max = 22.43, 23.93
R_EARTH = 6371.0 


def generate_step(exponent):
    """Generate a power-law distributed random variable."""
    return (np.random.pareto(exponent) + 1)

def get_initial_location():
    """Generate a random starting point within the study area."""
    return [random.uniform(lon_min, lon_max), random.uniform(lat_min, lat_max)]

def get_destination(lon1, lat1, d, angle_rad):
    """
    Calculate destination coordinates.
    """
    # Latitude change (approximate as arc length)
    delta_lat = (d * math.sin(angle_rad)) / R_EARTH
    lat2 = lat1 + math.degrees(delta_lat)
    
    # Longitude change (scaled by cos of latitude)
    avg_lat_rad = math.radians((lat1 + lat2) / 2)
    delta_lon = (d * math.cos(angle_rad)) / (R_EARTH * math.cos(avg_lat_rad))
    lon2 = lon1 + math.degrees(delta_lon)
    return [lon2, lat2]

def func_p(S, rho_val, gamma_val):
    return rho_val * (S**(-gamma_val))


def get_cluster_stats(fre_list, labels):
    """Calculate frequency of each cluster and return sorted list."""
    num_clusters = int(np.max(labels) + 1)
    c_fre = np.zeros(num_clusters)
    for f, l in zip(fre_list, labels):
        if l != -1:
            c_fre[l] += f
    # Return (frequency, cluster_id) sorted by frequency descending
    sorted_clusters = sorted([(val, i) for i, val in enumerate(c_fre)], key=lambda x: x[0], reverse=True)
    return sorted_clusters, c_fre


input_dir = r'D:\results\zxy\new_result\guangzhou\cluster_feature'
location_dir = r'D:\results\zxy\new_result\guangzhou\pattern'
output_path = r'D:\results\zxy\new_result\cepr_guangzhou\p'

if not os.path.exists(output_path):
    os.makedirs(output_path)

filenames = os.listdir(input_dir)

for user_id, filename in enumerate(filenames):
    print(f"Simulating User {user_id} (Total {len(filenames)})")
    
    while True: # Retry loop for each user
        empirical_data = pd.read_csv(os.path.join(location_dir, filename))
        max_visit = len(empirical_data)
        save_file = os.path.join(output_path, f"{user_id}.txt")
        
        # Initialization
        start_time = time.time()
        visit_count = 0
        S = 1
        current_time = generate_step(beta)
        curr_loc = get_initial_location()
        history = [curr_loc]
        fre = [1]
        i_node = curr_loc 
        flag = 'continue'

        # --- Single open session for the entire trajectory ---
        with open(save_file, 'w') as mfp:
            # Write initial location
            mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (0, current_time, curr_loc[0], curr_loc[1]))
            
            # Phase 1: Warming up until history size > 2
            while len(history) <= 2:
                p_new = func_p(S, rho, gamma)
                t_arr, current_time = current_time, current_time + generate_step(beta)
                t_lv = current_time
                
                if random.random() < p_new: # Explore
                    j_node = [0, 0]
                    while j_node in history or j_node == [0, 0]:
                        j_node = get_destination(i_node[0], i_node[1], generate_step(alpha), 2 * math.pi * random.random())
                    mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (t_arr, t_lv, j_node[0], j_node[1]))
                    history.append(j_node); fre.append(1); S += 1
                else: # Return
                    j_node = random.choices(history, weights=fre, k=1)[0]
                    mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (t_arr, t_lv, j_node[0], j_node[1]))
                    fre[history.index(j_node)] += 1
                
                i_node = j_node
                visit_count += 1
                
            # Phase 2: Cluster-based EPR Mechanism
            #Set simulation loops based on waiting time/number of visited locations/number of visits.
            while visit_count < max_visit:
                # Timeout check
                if time.time() - start_time >= 240:
                    flag = 'finish'
                    break

                p_new = func_p(S, rho, gamma)
                t_arr, current_time = current_time, current_time + generate_step(beta)
                t_lv = current_time
                
                # Perform HDBSCAN on current movement history
                hdb = hdbscan.HDBSCAN(min_cluster_size=3, cluster_selection_epsilon=0.01, prediction_data=True).fit(history)
                labels = hdb.labels_
                num_c = int(np.max(labels) + 1)
                
                if num_c == 0:
                    # Fallback to standard EPR if no clusters are formed
                    if random.random() < p_new:
                        j_node = [0, 0]
                        while j_node in history or j_node == [0, 0]:
                            j_node = get_destination(i_node[0], i_node[1], generate_step(alpha), 2 * math.pi * random.random())
                        history.append(j_node); fre.append(1); S += 1
                    else:
                        j_node = random.choices(history, weights=fre, k=1)[0]
                        fre[history.index(j_node)] += 1
                
                elif random.random() < p_new:
                    # Explore: Random vs In-cluster
                    p_rand_expl = func_p(S, 1, tau)
                    if p_rand_expl > 1 or random.random() < p_rand_expl:
                        # Random Exploration outside clusters
                        j_node = [0, 0]
                        while True:
                            if time.time() - start_time >= 240:
                                flag = 'finish'; break
                            j_node = get_destination(i_node[0], i_node[1], generate_step(alpha), 2 * math.pi * random.random())
                            if j_node not in history and hdbscan.approximate_predict(hdb, np.array([j_node]))[0][0] == -1:
                                break
                        if flag == 'finish': break
                        history.append(j_node); fre.append(1); S += 1
                    else:
                        # Explore within a selected cluster
                        sorted_c, _ = get_cluster_stats(fre, labels)
                        prob_k = omega / ((S + 1)**epsilon)
                        weights = [math.exp(prob_k * (r + 1)) for r in range(len(sorted_c))]
                        target_c = random.choices([item[1] for item in sorted_c], weights=weights, k=1)[0]
                        
                        j_node = [0, 0]
                        while True:
                            if time.time() - start_time >= 240:
                                flag = 'finish'; break
                            j_node = get_destination(i_node[0], i_node[1], generate_step(alpha), 2 * math.pi * random.random())
                            if j_node not in history and hdbscan.approximate_predict(hdb, np.array([j_node]))[0][0] == target_c:
                                break
                        if flag == 'finish': break
                        history.append(j_node); fre.append(1); S += 1
                else:
                    # Preferential Return to clusters
                    _, c_fre_array = get_cluster_stats(fre, labels)
                    target_c = random.choices(range(len(c_fre_array)), weights=c_fre_array, k=1)[0]
                    
                    # Return to a specific location within the chosen cluster
                    locs_in_c = [history[idx] for idx, l in enumerate(labels) if l == target_c]
                    loc_fres = [fre[idx] for idx, l in enumerate(labels) if l == target_c]
                    j_node = random.choices(locs_in_c, weights=loc_fres, k=1)[0]
                    fre[history.index(j_node)] += 1
                
                mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (t_arr, t_lv, j_node[0], j_node[1]))
                i_node = j_node
                visit_count += 1
        
        # Post-simulation check: decide whether to keep or retry
        if flag == 'continue':
            break # Successfully completed
        else:
            if os.path.exists(save_file):
                os.remove(save_file)
            print(f"User {user_id} session timeout or failed, retrying...")
            continue 

print("Simulation finished.")