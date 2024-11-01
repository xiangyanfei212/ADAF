Variables included in a sample file:

| Variable | Decription | Dimension |
| ----------- | ----------- | ----------- |
| z      | Topography, normalized      | [lat, lon] |
| rtma_t | T2M from RTMA, normalized   | [lat, lon] |
| rtma_q | Q from RTMA, normalized     | [lat, lon] |
| rtma_u10 | U10 from RTMA, normalized   | [lat, lon] |
| rtma_v10 | V10 from RTMA, normalized   | [lat, lon] |
| sta_t | T2M from station's observation, 0 means non-station, normalized | [obs_time_window, lat, lon] |
| sta_q | Q from station's observation, 0 means non-station, normalized   | [obs_time_window, lat, lon] |
| sta_u10 | U10 from station's observation, 0 means non-station, normalized | [obs_time_window, lat, lon] |
| sta_v10 | V10 from station's observation, 0 means non-station, normalized | [obs_time_window, lat, lon] |
| CMI02 | ABI Band 2: visible (red), normalized | [obs_time_window, lat, lon] |
| CMI07 | ABI Band 7: shortwave infrared, normalized | [obs_time_window, lat, lon] |
| CMI07 | ABI Band 10: low-level water vapor, normalized | [obs_time_window, lat, lon] |
| CMI14 | ABI Bands 14: longwave infrared, normalized | [obs_time_window, lat, lon] |
| hrrr_t | T2M from HRRR 1-hour forecast | [lat, lon]               |
| hrrr_q | Q from HRRR 1-hour forecast | [lat, lon]               |
| hrrr_u_10 | U10 from HRRR 1-hour forecast | [lat, lon]               |
| hrrr_v_10 | V10 from HRRR 1-hour forecast | [lat, lon]               |
