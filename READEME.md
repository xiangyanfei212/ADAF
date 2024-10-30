# ADAF

This repository contains the code used for "ADAF: An Artificial Intelligence Framework for Data Assimilation for Weather Forecasting" [paper]

(Abstract)
The forecasting skill of numerical weather prediction (NWP) models is critically dependent on the accuracy of the initial conditions given by data assimilation (DA). However, vast observations, high-dimensional state spaces, and complex mathematical operations in traditional DA methods make them computationally costly. We provide an AI-based data assimilation framework (ADAF) in this study to integrate multisource observations with high-resolution model simulation to produce km-scale analysis. This study is a first use of deep learning for the integration of satellite images with sparse surface weather data. ADAF is applied to four near-surface variables in the Continental United States (CONUS). With gains ranging from 16% to 33%, the findings show ADAF beats the High Resolution Rapid Refresh Data Assimilation System (HRRRDAS). According to the sensitivity study, ADAF can efficiently provide high-quality analysis even under very sparse surface observations and low-accuracy backgrounds.  ADAF also has the ability to reconstruct the condition of extreme storms. Operating weather forecasting might benefit ADAF, which can quickly absorb large real-world observations in a 3-hour window using less than 2 seconds on an AMD MI200 GPU.

[Figure: Overall framework]

## Data links
	- Pre-processes Training Data
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
