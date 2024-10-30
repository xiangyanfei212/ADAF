![image](https://github.com/user-attachments/assets/48d77575-4c61-4346-bf16-4245474b3dbc)# ADAF

This repository contains the code used for "ADAF: An Artificial Intelligence Data Assimilation Framework for Weather Forecasting"

## Abstract
The forecasting skill of numerical weather prediction (NWP) models critically depends on the accurate initial conditions, also known as analysis, provided by data assimilation (DA).
Traditional DA methods often face a trade-off between computational cost and accuracy due to complex linear algebra computations and the high dimensionality of the model, especially in nonlinear systems. Moreover, processing massive data in real-time requires substantial computational resources.
To address this, we introduce an artificial intelligence-based data assimilation framework (ADAF) to generate high-quality kilometer-scale analysis. 
This study is the pioneering work using real-world observations from varied locations and multiple sources to verify the AI method's efficacy in DA, including sparse surface weather observations and satellite imagery.
We implemented ADAF for four near-surface variables in the Contiguous United States (CONUS). 
The results demonstrate that ADAF outperforms the High Resolution Rapid Refresh Data Assimilation System (HRRRDAS) in accuracy by 16\% to 33\%, and is able to reconstruct extreme events, such as the wind field of tropical cyclones. 
Sensitivity experiments reveal that ADAF can generate high-quality analysis even with low-accuracy backgrounds and extremely sparse surface observations.  
ADAF can assimilate massive observations within a three-hour window at low computational cost, taking about two seconds on an AMD MI200 graphics processing unit (GPU). 
ADAF has been shown to be efficient and effective in real-world DA, underscoring its potential role in operational weather forecasting.

![Figure: Overall framework](/assets/framework.png)


## Data
- Pre-processed data
  The pre-proccesd data consists of input-target pairs. The inputs include surface weather observations within a 3-hour window, GOES-16 satellite imagery within a 3-hour window, HRRR forecast, and topography. The target is a combination of RTMA and surface weather observations.
  |||||||
  |||||||
- Trained model weights
- Pre-computed normalization statistics

## Inference
In order to run ADAF in inference mode you will need to have the following files on hand.

1. The path to the out of training sample file.

2. The inference script
You can modify the script to use a different h5 file that you processed yourself after downloading the raw data from Zenodo.

3. The model weights hosted at Trained Model Weights。

4. The Pre-computed normalization statistics

5. The configuration file (./config/experiment.yaml)

Once you have all the file listed above you should be ready to go.

Run inference using
'''
export CUDA_VISIBLE_DEVICES='0'

seed=0
exp_dir='./exp/'              # directory to save prediction 
net_config='EncDec'           # Network configuration
test_data_path='./data/test'  # path to test data
inference_time_file='./utils/test_date.txt'  # Generate analysis at these time
hold_out_obs_ratio=0.3        # the ratio of surface observation be inputted to model

nohup python -u 03_inference.py \
    --seed=${seed} \
    --exp_dir=${exp_dir} \
    --test_data_path=${test_data_path} \
    --net_config=${net_config} \
    --hold_out_obs_ratio=${hold_out_obs_ratio} \
    --inference_by_time=False \
    --inference_time_file=${inference_time_file} \
    --overwrite=True \
    > inference.log 2>&1 &

'''


## Links

[Link for Pre-processed Data]

[Link for Pre-processed trained model weights]

[Link for pre-computed normalization statistics]










## Inference
