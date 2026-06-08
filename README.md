This repository contains the official code and sample data implementation for the paper:

**The cluster effect: How spatial grouping governs human mobility**
*Xinyuan Zhang<sup>a,b</sup>, Qi Wang<sup>c</sup>, Bo Huang<sup>b</sup>, Dongping Fang<sup>a</sup>, Nan Li<sup>a,*</sup>*

<sup>a</sup> *Department of Construction Management, Tsinghua University, Beijing 100084, China* <sup>b</sup> *Department of Geography, The University of Hong Kong, Hong Kong, China* <sup>c</sup> *Department of Civil and Environmental Engineering, Northeastern University, Boston MA 02115, United States* <sup>*</sup> *Corresponding author: [nanli@tsinghua.edu.cn]*

---

## Repository Structure

The repository is organized into the following key directories:

```text
├── code/
│   ├── EPR_model.py                     # Baseline Exploration and Return (EPR) mobility model
│   ├── cluster_based_model.py           # The proposed cluster-based human mobility model
│   ├── inter&intra_mobility_pattern.py  # Analysis of mobility pattern within and between clusters
│   ├── location_cluster_correlation.py  # Correlation between locations and clusters
│   ├── model_parameters.py              # Configuration of parameters
│   ├── post_processing_trajectory_data.py # Post-processing pipeline for mobility patterns
│   ├── scale_invariance.py              # Validation of spatial scale-invariance
│   └── visitation_spatial_pattern.py   # Spatial patterns of visitation frequencies
├── data/ 
└── README.md

1. code/
This directory contains the core implementation of our mobility models and key analyses. 

2. data/
Contains preprocessed intermediate matrices (in .npy/.txt format) for empirical validation.

📊 Data Availability Statement
Due to strict Non-Disclosure Agreements (NDAs) and privacy regulations, we cannot disclose the raw data publicly in this repository. To ensure the reproducibility of our research findings, we share preprocessed intermediate matrices (found in the data/ folder) used for validating the key results. However, because some outputs are massive and exceed GitHub's file storage thresholds, only a subset of the sample data is uploaded here for demonstration purposes.
If you require the full set of intermediate validation results or wish to request collaborative access for academic verification, please contact the corresponding author.
