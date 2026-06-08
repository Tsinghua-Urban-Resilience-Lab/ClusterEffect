# -*- coding: utf-8 -*-
"""
The cluster effect: How spatial grouping governs human mobility
Individual Mobility Simulation using the EPR Model

@author: Xinyuan, Zhang
"""

import os
import math
import random
import numpy as np
import pandas as pd
#Parameter Setting
#For Guangzhou: alpha,beta,rho,gamma,lon_min,lon_max,lat_min,lat_max=0.615,0.618,0.571,0.283,112.95, 114.05,22.43, 23.93
#For Houston:alpha,beta,rho,gamma,lon_min,lon_max,lat_min,lat_max=0.415,0.800,0.618,0.264,-96.13,-94.69,29.37,30.29

# --- Model Parameters ---
alpha = 0.615   
beta = 0.618    
rho = 0.571    
gamma = 0.283


# Spatial Boundaries (Guangzhou)
lon_min, lon_max = 112.95, 114.05
lat_min, lat_max = 22.43, 23.93
R_EARTH = 6371.0 

# --- Helper Functions ---

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

def func_p(S, rho_, gamma_):
    """Exploration probability P_new = rho * S^(-gamma)."""
    return rho_ * (S**(-gamma_))

# --- Simulation Execution ---

# Path Configuration
input_dir = r'D:\results\zxy\new_result\guangzhou\cluster_feature'
location_dir = r'D:\results\zxy\new_result\guangzhou\pattern'
output_path = r'D:\results\zxy\new_result\epr_guangzhou\p'

if not os.path.exists(output_path):
    os.makedirs(output_path)

filenames = os.listdir(input_dir)

for user_id, filename in enumerate(filenames):
    # Match simulated visit count with empirical data
    empirical_data = pd.read_csv(os.path.join(location_dir, filename))
    max_visit = len(empirical_data)
    
    print(f"Simulating User {user_id} (Total {len(filenames)})")
    
    # Simulation Initial State
    save_file = os.path.join(output_path, f"{user_id}.txt")
    
    with open(save_file, 'w') as mfp:
        S = 1            # Distinct locations visited
        visit_count = 0
        current_time = generate_step(beta)
        
        # Initial Position
        curr_loc = get_initial_location()
        mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (0, current_time, curr_loc[0], curr_loc[1]))
        
        history = [curr_loc]
        freq = [1]
        
        #Set simulation loops based on waiting time/number of visited locations/number of visits.
        while visit_count < max_visit:
            p_new = func_p(S, rho, gamma)
            t_arrive = current_time
            current_time += generate_step(beta)
            t_leave = current_time
            
            # Decision: Explore new or Return to old
            if random.random() < p_new:
                # Explore Strategy
                new_loc = [0, 0]
                # Avoid collision with history using epsilon comparison
                while any(np.allclose(new_loc, h, atol=1e-7) for h in history) or new_loc == [0, 0]:
                    delta_r = generate_step(alpha)
                    angle = 2 * math.pi * random.random()
                    new_loc = get_destination(curr_loc[0], curr_loc[1], delta_r, angle)
                
                curr_loc = new_loc
                history.append(curr_loc)
                freq.append(1)
                S += 1
            else:
                # Return Strategy (Preferential Return)
                prev_prob = [f / sum(freq) for f in freq]
                # Efficiently pick from history
                curr_loc = random.choices(history, weights=prev_prob, k=1)[0]
                loc_idx = history.index(curr_loc)
                freq[loc_idx] += 1
            
            mfp.write('%.5f,%.5f,%.5f,%.5f\n' % (t_arrive, t_leave, curr_loc[0], curr_loc[1]))
            visit_count += 1

print("EPR Simulation Phase Complete.")