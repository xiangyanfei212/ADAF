# ADAF

This repository contains the code used for "ADAF: An Artificial Intelligence Data Assimilation Framework for Weather Forecasting"

(Abstract)
The forecasting skill of numerical weather prediction (NWP) models critically depends on the accurate initial conditions, also known as analysis, provided by data assimilation (DA).
Traditional DA methods often face a trade-off between computational cost and accuracy due to complex linear algebra computations and the high dimensionality of the model, especially in nonlinear systems. Moreover, processing massive data in real-time requires substantial computational resources.
To address this, we introduce an artificial intelligence-based data assimilation framework (ADAF) to generate high-quality kilometer-scale analysis. 
This study is the pioneering work using real-world observations from varied locations and multiple sources to verify the AI method's efficacy in DA, including sparse surface weather observations and satellite imagery.
We implemented ADAF for four near-surface variables in the Contiguous United States (CONUS). 
The results demonstrate that ADAF outperforms the High Resolution Rapid Refresh Data Assimilation System (HRRRDAS) in accuracy by 16\% to 33\%, and is able to reconstruct extreme events, such as the wind field of tropical cyclones. 
Sensitivity experiments reveal that ADAF can generate high-quality analysis even with low-accuracy backgrounds and extremely sparse surface observations.  
ADAF can assimilate massive observations within a three-hour window at low computational cost, taking about two seconds on an AMD MI200 graphics processing unit (GPU). 
ADAF has been shown to be efficient and effective in real-world DA, underscoring its potential role in operational weather forecasting.

![Figure: Overall framework]("/assets/framework.png")


## Data links
- Pre-processed Data
- Trained Model Weights

## Training 

The training dataset of the AI model consists of input-target pairs. The inputs include surface weather observations within a 3-hour window, GOES-16 satellite imagery within a 3-hour window, HRRR forecast, and topography. The target is RTMA.
All data were regularized to grids of size $512 \times 1280$ with a spatial resolution of $0.05 \times 0.05 ^\circ$. 
Table~\ref{tab:datasets_summary} provides a summary of the input and target datasets used in our study.



For convenience it is available to part of data via Globus at the following link.

[Link for Pre-processed Training Data]

The data directory is organized as follows:

Data
│   README.md
└───train
│   │   1979.h5
│   
└───test
│   │   2016.h5
│
└───out_of_sample
│   │   2018.h5
│
└───static
│   │   orography.h5
│   |    statistic_norm_value.csv

Training configurations can be set up in config/  yaml. The following paths need to be set by the user. These paths should point to the data and stats you downloaded in the steps above:






## Inference
