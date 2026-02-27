mode='valid' # train, valid, test
lead_time=1 # background's lead time

# These paths need be channged to your data directory
index_file='./samples_hour_0_6_12_18/'${mode}'_index.csv'
obs_dir='./station-data/hourly/obs_grid_station_us_761_1591_more_q_ws_p_wg_td_ra/'
rtma_dir='./Analysis/RTMA/rtma2p5_grid/grid_data_5km_v2/daily/35variables_topography_norm/'
hrrr_dir='./HRRR/grid_data_5km_version3/hrrr/prs/'
satelite_dir='./OBS/Satellite/grid/daily_nc/lat20-55_lon225-300_5km_1h_v2/C2-7-10-14_dropnan_norm/'
topography_file='./Analysis/RTMA/rtma2p5_grid/grid_data_5km_v2/daily/35variables_topography_norm/20230929.nc'
stats_file='./stats_variable_wise_ignore_extreme.csv'

hold_out_obs_ratio=0.1 # ratio of hold-out in-situ observation
obs_time_window=3 # Time window of observations

out_dir='./samples_hour_0_6_12_18/lead_'${lead_time}'/'${mode}

nohup python -u sample_generate.py \
    --index_file=${index_file} \
    --obs_dir=${obs_dir} \
    --satelite_dir=${satelite_dir} \
    --hrrr_dir=${hrrr_dir} \
    --rtma_dir=${rtma_dir} \
    --topography_file=${topography_file} \
    --stats_file=${stats_file} \
    --out_dir=${out_dir} \
    --hold_out_obs_ratio=${hold_out_obs_ratio} \
    --obs_time_window=${obs_time_window} \
    --lead_time=${lead_time} \
    --hrrr_variables q sp t u_10 v_10 \
    --rtma_variables q sp t u10 v10 \
    --obs_variables q p t u10 v10 \
    --satelite_variables CMI07 CMI10 CMI14 CMI02 \
    --lat_range 24.675 50.275 \
    --lon_range 231.975 295.975 \
    --overwrite=True > logs/gen_sample_${mode}_lead_${lead_time}.log 2>&1 & 
