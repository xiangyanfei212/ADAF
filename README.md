# ADAF

This repository contains the code used for "ADAF: An Artificial Intelligence Data Assimilation Framework for Weather Forecasting"

## Abstract
The forecasting skill of numerical weather prediction (NWP) models critically depends on the accurate initial conditions, also known as analysis, provided by data assimilation (DA).
Traditional DA methods often face a trade-off between computational cost and accuracy due to complex linear algebra computations and the high dimensionality of the model, especially in nonlinear systems. Moreover, processing massive data in real-time requires substantial computational resources. To address this, we introduce an artificial intelligence-based data assimilation framework (ADAF) to generate high-quality kilometer-scale analysis. This study is the pioneering work using real-world observations from varied locations and multiple sources to verify the AI method's efficacy in DA, including sparse surface weather observations and satellite imagery. We implemented ADAF for four near-surface variables in the Contiguous United States (CONUS). The results demonstrate that ADAF outperforms the High Resolution Rapid Refresh Data Assimilation System (HRRRDAS) in accuracy by 16\% to 33\%, and is able to reconstruct extreme events, such as the wind field of tropical cyclones. Sensitivity experiments reveal that ADAF can generate high-quality analysis even with low-accuracy backgrounds and extremely sparse surface observations. ADAF can assimilate massive observations within a three-hour window at low computational cost, taking about two seconds on an AMD MI200 graphics processing unit (GPU). ADAF has been shown to be efficient and effective in real-world DA, underscoring its potential role in operational weather forecasting.

![Figure: Overall framework](/assets/framework.png)


## Data
- Pre-processed data

  [Link for Pre-processed Data - Zenodo Download Link](https://zenodo.org/records/14020879)

  The pre-proccesd data consists of input-target pairs. The inputs include surface weather observations within a 3-hour window, GOES-16 satellite imagery within a 3-hour window, HRRR forecast, and topography. The target is a combination of RTMA and surface weather observations. The table below summarizes the input and target datasets utilized in this study. All data were regularized to grids of size 512 $\times$ 1280 with a spatial resolution of 0.05 $\times$ 0.05 $^\circ$. 
	<table>
		<tr>
		    <td></td>
		    <td><b>Dataset</b></td>
		    <td><b>Source</b></td>
		    <td><b>Time window</b></td>
		    <td><b>Variables/Bands</b></td>
		</tr>
		<tr>
		    <td rowspan="4"><b>Input</b></td>
		    <td>Surface weather observations</td>
		    <td>WeatherReal-Synoptic (Jin et al., 2024)</td>
		    <td>3 hours</td>
		    <td>Q, T2M, U10, V10</td>  
		</tr>
	 	<tr>
		    <td>Satellite imagery</td>
                    <td>GOES-16 (Tan et al., 2019)</td>
		    <td>3 hours</td>
		    <td>0.64, 3.9, 7.3, 11.2 $\mu m$</td>  
		</tr>
	 	<tr>
		    <td>Background</td>
 		    <td>HRRR forecast (Dowell et al., 2022)</td>
		    <td>N/A</td>
		    <td>Q, T2M, U10, V10</td>  
		</tr>
	 	<tr>
		    <td>Topography</td>
		    <td>ERA5 (Hersbach et al., 2019)</td>
		    <td>N/A</td>
		    <td>Geopotential</td>  
		</tr>
	 	<tr>
		    <td rowspan="2"><b>Target</b></td>
		    <td>Analysis</td>
		    <td>RTMA (Pondeca et al., 2011)</td>
		    <td>N/A</td>
		    <td>Q, T2M, U10, V10</td>  
		</tr>
	 	<tr>
		    <td>Surface weather observations</td>
		    <td>WeatherReal-Synoptic (Jin et al., 2024)</td>
		    <td>N/A</td>
		    <td>Q, T2M, U10, V10</td> 
		</tr>
	</table>

	
- Pre-computed normalization statistics

  [Link for pre-computed normalization statistics- Zenodo Download Link](https://zenodo.org/records/14020879).

  If you are utilizing the pre-trained model weights that we provided, it is crucial that you utilize of the given statistics as these were used during model training. The learned model weights complement the normalizing statistics exactly.
  
  The data directory of pre-processed data and pre-computed normalization statistics is organized as follows:
  ```
  data
  │   README.md
  └───test
  │   │   2022-10-01_00.nc
  │   │   2022-10-02_06.nc
  │   │   2022-10-03_12.nc
  │   │   ...
  │   │   2023-10-31_00.nc
  └───stats.csv
  ```

- Trained model weights

  [Link for trained model weights - Zenodo Download Link](https://zenodo.org/records/14020879)

  ```
  model_weights/
  │   model_trained.ckpt
  ```

## Inference
In order to run ADAF in inference mode you will need to have the following files on hand.

1. The path to the test sample file. (./data/test/)

2. The inference script (inference.py)

3. The model weights hosted at Trained Model Weights。 (./model_weights/model_trained.ckpt)

4. The pre-computed normalization statistics (./data/stats.csv)

5. The configuration file (./config/experiment.yaml)

Once you have all the file listed above you should be ready to go.

An example launch script for inference is provided. 
```shell
export CUDA_VISIBLE_DEVICES='0'

nohup python -u inference.py \
    --seed=0 \
    --exp_dir='./exp/' \ 		# directory to save prediction 
    --test_data_path='./data/test' \ 	# path to test data
    --net_config='EncDec' \		# network configuration
    --hold_out_obs_ratio=0.3 \		# the ratio of surface observations to be fed into the model
    > inference.log 2>&1 &

```



## References

```
1. Jin, W. et al. WeatherReal: A Benchmark Based on In-Situ Observations for Evaluating Weather Models. (2024).
2. Dowell, D. et al. The High-Resolution Rapid Refresh (HRRR): An Hourly Updating Convection-Allowing Forecast Model. Part I: Motivation and System Description. Weather and Forecasting 37, (2022).
3. Tan, B., Dellomo, J., Wolfe, R. & Reth, A. GOES-16 and GOES-17 ABI INR assessment. in Earth Observing Systems XXIV vol. 11127 290–301 (SPIE, 2019).
4. Hersbach, H. et al. ERA5 monthly averaged data on single levels from 1979 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS) 10, 252–266 (2019).
5. Pondeca, M. S. F. V. D. et al. The Real-Time Mesoscale Analysis at NOAA’s National Centers for Environmental Prediction: Current Status and Development. Weather and Forecasting 26, 593–612 (2011).
```
