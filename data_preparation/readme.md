# Sample Generation for Model Training/Evaluation/Test

## Step 1: Prepare the Data

The dataset consists of input-target pairs. The **inputs** include:

* Surface weather observations (3-hour window)
* GOES-16 satellite imagery (3-hour window)
* HRRR forecast data
* Topography data

The **target** is a combination of RTMA and surface weather observations. All data were normalized to a grid of size 512 × 1280 with a spatial resolution of 0.05 × 0.05°.

### Exmples for data paths:

```bash
obs_dir='/wxforecasting/daily-forecast/station-data/hourly/obs_grid_station_us_761_1591_more_q_ws_p_wg_td_ra/'
rtma_dir='/wxforecasting3/OMG-HD/Analysis/RTMA/rtma2p5_grid/grid_data_5km_v2/daily/35variables_topography_norm/'
hrrr_dir='/wxforecasting/users/zuliang/data/HRRR/grid_data_5km_version3/hrrr/prs/'
satelite_dir='/wxforecasting3/OMG-HD/OBS/Satellite/grid/daily_nc/lat20-55_lon225-300_5km_1h_v2/C2-7-10-14_dropnan_norm/'
topography_file='/wxforecasting3/OMG-HD/Analysis/RTMA/rtma2p5_grid/grid_data_5km_v2/daily/35variables_topography_norm/20230929.nc'
stats_file='/wxforecasting3/users/v-yanfei/samples_v12_hour_0_6_12_18/lead_1/stats_variable_wise_ignore_extreme.csv'
```

### Data Summary:

| **Dataset**                  | **Source**                  | **Time Window** | **Variables/Bands**     | **Code for pre-processing** |
| ---------------------------- | --------------------------- | --------------- | ----------------------- | ------------------- |
| Surface weather observations | WeatherReal-Synoptic (2024) | 3 hours         | Q, T2M, U10, V10        | code [TODO]         |
| Satellite imagery            | GOES-16                     | 3 hours         | 0.64, 3.9, 7.3, 11.2 μm | code [TODO]         |
| HRRR forecast                | HRRR                        | N/A             | Q, T2M, U10, V10        | code [TODO]         |
| RTMA analysis                | RTMA                        | N/A             | Q, T2M, U10, V10        | code [TODO]         |
| Topography                   | ERA5                        | N/A             | Geopotential            | code [TODO]         |

---

## Step 2: Generate Time Sequences for Training/Evaluation/Testing

Run the following script to generate time sequences:

```bash
python time_index_generate.py
```

---

## Step 3: Generate Samples

Run the script `run.sh` to generate samples for training, evaluation, and testing. Make sure the data paths (e.g., `obs_dir`, `rtma_dir`, `hrrr_dir`, `satelite_dir`, `topography_file`, `stats_file`) are correctly set.

* [Pre-computed normalization statistics](https://zenodo.org/records/14020879) (Zenodo Download Link)



