import os
import datetime
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic
from str2bool import str2bool


def get_grid_alt(topo_file, lon_range, lat_range):
    ds_topo = xr.open_dataset(topo_file)
    alt = ds_topo["z"].sel(lon=lon_range, lat=lat_range).values[0]
    return alt


def reverse_norm(
    data,
    variable_names,
    stats_file,
):

    stats = pd.read_csv(stats_file, index_col=0)

    # for vi, var in enumerate(params.field_tar_vars):
    for vi, var in enumerate(variable_names):
        if vi == 0:
            field_tar_stats = stats[stats["variable"].isin([var])]
        else:
            field_tar_stats = pd.concat(
                [field_tar_stats, stats[stats["variable"].isin([var])]]
            )
    ic(field_tar_stats)

    vmin = field_tar_stats["min"]
    vmax = field_tar_stats["max"]

    if len(data.shape) == 4:
        vmin = np.array(vmin)[:, np.newaxis, np.newaxis, np.newaxis]
        vmin = np.repeat(vmin, data.shape[1], axis=1)
        vmin = np.repeat(vmin, data.shape[2], axis=2)
        vmin = np.repeat(vmin, data.shape[3], axis=3)
        vmin = np.squeeze(vmin)

        vmax = np.array(vmax)[:, np.newaxis, np.newaxis, np.newaxis]
        vmax = np.repeat(vmax, data.shape[1], axis=1)
        vmax = np.repeat(vmax, data.shape[2], axis=2)
        vmax = np.repeat(vmax, data.shape[3], axis=3)
        vmax = np.squeeze(vmax)

    if len(data.shape) == 3:
        vmin = np.array(vmin)[:, np.newaxis, np.newaxis]
        vmin = np.repeat(vmin, data.shape[1], axis=1)
        vmin = np.repeat(vmin, data.shape[2], axis=2)
        vmin = np.squeeze(vmin)

        vmax = np.array(vmax)[:, np.newaxis, np.newaxis]
        vmax = np.repeat(vmax, data.shape[1], axis=1)
        vmax = np.repeat(vmax, data.shape[2], axis=2)
        vmax = np.squeeze(vmax)

    if len(data.shape) == 2:
        vmin = np.array(vmin)[np.newaxis, np.newaxis]
        vmin = np.repeat(vmin, data.shape[0], axis=0)
        vmin = np.repeat(vmin, data.shape[1], axis=1)
        vmin = np.squeeze(vmin)

        vmax = np.array(vmax)[np.newaxis, np.newaxis]
        vmax = np.repeat(vmax, data.shape[0], axis=0)
        vmax = np.repeat(vmax, data.shape[1], axis=1)
        vmax = np.squeeze(vmax)

    data = (data + 1) * (vmax - vmin) / 2 + vmin

    return data


def min_max_norm_ignore_extreme_fill_nan(
    data,
    variable_names,
    stats_file,
):

    stats = pd.read_csv(stats_file, index_col=0)

    # for vi, var in enumerate(params.field_tar_vars):
    for vi, var in enumerate(variable_names):
        if vi == 0:
            field_tar_stats = stats[stats["variable"].isin([var])]
        else:
            field_tar_stats = pd.concat(
                [field_tar_stats, stats[stats["variable"].isin([var])]]
            )
    ic(field_tar_stats)

    vmin = field_tar_stats["min"]
    vmax = field_tar_stats["max"]

    if data.ndim == 4:
        vmax = np.array(vmax)[:, np.newaxis, np.newaxis, np.newaxis]
        vmin = np.array(vmin)[:, np.newaxis, np.newaxis, np.newaxis]
        vmax = np.repeat(vmax, data.shape[1], axis=1)
        vmax = np.repeat(vmax, data.shape[2], axis=2)
        vmax = np.repeat(vmax, data.shape[3], axis=3)
        vmin = np.repeat(vmin, data.shape[1], axis=1)
        vmin = np.repeat(vmin, data.shape[2], axis=2)
        vmin = np.repeat(vmin, data.shape[3], axis=3)
    elif data.ndim == 3:
        vmax = np.array(vmax)[:, np.newaxis, np.newaxis]
        vmin = np.array(vmin)[:, np.newaxis, np.newaxis]
        vmax = np.repeat(vmax, data.shape[1], axis=1)
        vmax = np.repeat(vmax, data.shape[2], axis=2)
        vmin = np.repeat(vmin, data.shape[1], axis=1)
        vmin = np.repeat(vmin, data.shape[2], axis=2)

    data -= vmin
    data *= 2.0 / (vmax - vmin)
    data -= 1.0

    data = np.where(data > 1, 1, data)
    data = np.where(data < -1, -1, data)
    data = np.nan_to_num(data, nan=0)

    return data


def read_forecast_from_hrrr(
    variables: list,
    year: int,
    month: int,
    day: int,
    hour: int,
    lon_range: list,
    lat_range: list,
    lead_time: int,
    hrrr_dir: str,
):
    """
    find the specific lead-time forecast at (year, month, day, hour)
    """

    analysis_time = datetime.datetime(
        year=year, month=month, day=day, hour=hour)
    analysis_date = analysis_time.strftime("%Y%m%d")

    if hour >= 1:
        hrrr_date = analysis_date
        hrrr_file = os.path.join(
            hrrr_dir,
            hrrr_date,
            hrrr_date
            + f"T{str(hour-lead_time).zfill(2)}-\
            f{str(lead_time).zfill(2)}.nc",
        )

        if not os.path.exists(hrrr_file):
            print(f"{hrrr_file} not exits, skip!")
            hrrr_file = os.path.join(
                hrrr_dir,
                hrrr_date,
                hrrr_date
                + f"T{str(hour-lead_time-1).zfill(2)}-\
                f{str(lead_time+1).zfill(2)}.nc",
            )
            print(f"Use {hrrr_file}")

    else:  # hour == 0
        # preivous date
        hrrr_date = analysis_time - datetime.timedelta(days=1)
        hrrr_date = hrrr_date.strftime("%Y%m%d")
        hrrr_file = os.path.join(
            hrrr_dir,
            hrrr_date,
            hrrr_date
            + f"T{str(24-lead_time).zfill(2)}-\
            f{str(lead_time).zfill(2)}.nc",
        )

        if not os.path.exists(hrrr_file):
            print(f"{hrrr_file} not exits, skip!")
            hrrr_file = os.path.join(
                hrrr_dir,
                hrrr_date,
                hrrr_date
                + f"T{str(24-lead_time-1).zfill(2)}-\
                f{str(lead_time+1).zfill(2)}.nc",
            )
            print(f"Use {hrrr_file}")

    print(f"Reading hrrr from: {hrrr_file}")
    try:
        ds = xr.open_dataset(hrrr_file)
    except NameError:
        print(f"Something wrong when reading {hrrr_file}")
        return False, 0, 0, 0

    ds = ds.sel(
        lon=slice(lon_range[0], lon_range[1]),
        lat=slice(lat_range[0], lat_range[1]),
    )
    ic(ds)
    lon = ds["lon"].values
    lat = ds["lat"].values

    data = np.squeeze(np.array(ds[variables].to_array()))

    for vi, v in enumerate(variables):
        data_var = data[vi]

        print(f"{v}: {np.min(data_var)}~{np.max(data_var)}")
        if v == "sp":
            uqc = len(np.where(data_var < 600)[0])
            print("sp:", data_var.shape)
            print(f"sp, out of range: {uqc}")

    return True, data, np.array(lon), np.array(lat)


def read_data_from_RTMA(
    rtma_file,
    variables,
    hour,
    lon_range,
    lat_range
):

    ds = xr.open_dataset(rtma_file)
    ds = ds.sel(
        lon=slice(lon_range[0], lon_range[1]),
        lat=slice(lat_range[0], lat_range[1]),
    )
    data = np.array(ds[variables].to_array())[:, hour]
    return data


def get_observation_in_time_window(
    # obs_file: str,
    start_time,
    end_time,
    obs_dir: str,
    variable_names: list,
    lon_range: list,
    lat_range: list,
    hold_out_ratio: float,  # 0 means not hold-out
):
    obs_file = os.path.join(
        obs_dir,
        end_time.strftime("%Y%m") + ".nc")
    print(f"obs_file: {obs_file}")
    ds = xr.open_dataset(obs_file)

    ds = ds.sel(
        time=slice(start_time, end_time),
        lon=slice(lon_range[0], lon_range[1]),
        lat=slice(lat_range[0], lat_range[1]),
    )
    # ic(ds)
    data = np.array(ds[variable_names].to_array())
    ic(data.shape)

    # %% Generate a mask
    lat_num = data[0, 0].shape[0]
    lon_num = data[0, 0].shape[1]

    # [lat, lon] -> [lat * lon]
    data_tw_begin = data[0, 0].reshape(-1)
    # ic(data_tw_begin.shape)

    obs_index = np.where(~np.isnan(data_tw_begin))[0]
    print(
        f"obs_index:{obs_index}, \
        range:{np.min(obs_index)}~{np.max(obs_index)}"
    )

    obs_num = len(obs_index)
    hold_out_num = int(obs_num * hold_out_ratio)
    print(f"obs_num: {obs_num}, hold_out_num: {hold_out_num}")

    np.random.shuffle(obs_index)
    # print(f'obs_index after shuffle:{obs_index}, \
    # range:{np.min(obs_index)}~{np.max(obs_index)}')

    hold_out_obs_index = obs_index[:hold_out_num]
    input_obs_index = obs_index[hold_out_num:]
    print(f"hold_out_obs_index: {hold_out_obs_index}")
    print(f"input_obs_index: {input_obs_index}")

    # Mask (lat, lon), hold_out obs=1, input obs = 0
    mask = np.zeros(data_tw_begin.shape)
    mask[hold_out_obs_index] = 1
    mask = mask.reshape([lat_num, lon_num])

    return data, mask


def read_data_from_satelite_in_time_window(
    satelite_dir: str,
    variable_names: list,
    obs_time_window: int,
    year: int,
    month: int,
    day: int,
    hour: int,
    lon_range: list,
    lat_range: list,
):

    if hour != 0:
        analysis_time = datetime.datetime(
            year=year, month=month, day=day, hour=hour)
        file_path = os.path.join(
            satelite_dir,
            analysis_time.strftime("%Y%m%d") + ".nc")

        if not os.path.exists(file_path):
            print(f"{file_path} not exits, skip!")
            return False, None, None, None

        print(f"Reading {file_path}")
        try:
            ds = xr.open_dataset(file_path)
        except NameError:
            print(f"Something wrong when reading {file_path}")
            return False, None, None, None
        ds = ds.sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[0], lat_range[1]),
        )
        lon = ds["lon"].values
        lat = ds["lat"].values
        # ic(ds)

        data = np.array(ds[variable_names].to_array())[
            :, hour - obs_time_window + 1: hour + 1
        ]
    else:
        analysis_time = datetime.datetime(
            year=year, month=month, day=day, hour=hour)
        file_path = os.path.join(
            satelite_dir,
            analysis_time.strftime("%Y%m%d") + ".nc")
        file_path_0 = os.path.join(
            satelite_dir,
            (analysis_time - datetime.timedelta(days=1)).strftime(
                "%Y%m%d") + ".nc",
        )

        if not os.path.exists(file_path):
            print(f"{file_path} not exits, skip!")
            return False, None, None, None

        if not os.path.exists(file_path_0):
            print(f"{file_path_0} not exits, skip!")
            return False, None, None, None

        print(f"Reading {file_path}")
        try:
            ds = xr.open_dataset(file_path)
        except NameError:
            print(f"Something wrong when reading {file_path}")
            return False, None, None, None
        ds = ds.sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[0], lat_range[1]),
        )
        lon = ds["lon"].values
        lat = ds["lat"].values
        data = np.array(ds[variable_names].to_array())[:, 0][
            :, np.newaxis, :, :]
        # ic(data.shape)

        try:
            ds_0 = xr.open_dataset(file_path_0)
        except NameError:
            print(f"Something wrong when reading {file_path}")
            return False, None, None, None
        ds_0 = ds_0.sel(
            lon=slice(lon_range[0], lon_range[1]),
            lat=slice(lat_range[0], lat_range[1]),
        )
        data_0 = np.array(ds_0[variable_names].to_array())[
            :, -obs_time_window + 1:]

        if data_0.shape[3] != data.shape[3]:
            return False, None, None, None
        else:
            data = np.concatenate([data_0, data], axis=1)

    return True, data, lat, lon


def generate_samples_hourly(
    analysis_year: int,
    analysis_month: int,
    analysis_day: int,
    analysis_hour: int,
    lon_range: list,
    lat_range: list,
    lead_time: int,
    obs_time_window: int,
    hrrr_variables: list,
    rtma_variables: list,
    obs_variables: list,
    satelite_variables: list,
    obs_dir: str,
    rtma_dir: str,
    hrrr_dir: str,
    satelite_dir: str,
    out_dir: str,
    topography_file: str,
    hold_out_obs_ratio: float,
    stats_file: str,
    overwrite=False,
):

    analysis_time = datetime.datetime(
        year=analysis_year,
        month=analysis_month,
        day=analysis_day,
        hour=analysis_hour,
    )
    print(f"analysis_time:{analysis_time}")
    analysis_date_str = analysis_time.strftime("%Y%m%d")
    analysis_time_str = analysis_time.strftime("%Y-%m-%d_%H")

    out_file = os.path.join(out_dir, f"{analysis_time_str}.nc")
    if (not overwrite) and os.path.exists(out_file):
        # if os.path.exists(out_file):
        print(f"{out_file} exists, skip")
        return

    # %% reading forecast data from hrrr
    print("Reading HRRR ...")
    exists_flag, hrrr_data, hrrr_lon, hrrr_lat = read_forecast_from_hrrr(
        hrrr_dir=hrrr_dir,
        year=analysis_year,
        month=analysis_month,
        day=analysis_day,
        hour=analysis_hour,
        lon_range=lon_range,
        lat_range=lat_range,
        variables=hrrr_variables,
        lead_time=lead_time,
    )
    if not exists_flag:
        return
    ic(hrrr_data.shape)

    # %% Read altitude
    print("\nReading altitude ...")
    topo = get_grid_alt(topography_file, hrrr_lon, hrrr_lat)
    ic(topo.shape, np.min(topo), np.max(topo))

    # %% Read RTMA
    print("\nReading RTMA ...")
    rtma_file = os.path.join(rtma_dir, f"{analysis_date_str}.nc")
    if not os.path.exists(rtma_file):
        print(f"{rtma_file} not exists, skip")
        return
    print(f"rtma_file: {rtma_file}")
    rtma_data = read_data_from_RTMA(
        rtma_file, rtma_variables, analysis_hour, lon_range, lat_range
    )
    # [#variable, lat, lon]
    ic(rtma_data.shape, np.min(rtma_data), np.max(rtma_data))

    rtma_data_unnorm = reverse_norm(
        rtma_data,
        ["rtma_" + var for var in rtma_variables],
        stats_file,
    )
    ic(rtma_data_unnorm.shape)

    # For observation quality control
    rtma_range = {}
    for vi, var in enumerate(rtma_variables):
        rtma_range[f"{var}_min"] = np.nanmin(rtma_data_unnorm[vi])
        rtma_range[f"{var}_max"] = np.nanmax(rtma_data_unnorm[vi])
    ic(rtma_range)

    # %% Observation
    print("\nReading Observation....")
    obs_start_time = analysis_time - datetime.timedelta(
        hours=obs_time_window - 1)
    obs_data, obs_mask = get_observation_in_time_window(
        obs_dir=obs_dir,
        variable_names=obs_variables,
        start_time=obs_start_time,
        end_time=analysis_time,
        lon_range=lon_range,
        lat_range=lat_range,
        hold_out_ratio=hold_out_obs_ratio,
    )
    ic(obs_data.shape, obs_mask.shape)
    if obs_data.shape[1] != obs_time_window:
        print("incomplete observation, skip!")
        return

    # Quality control for observation
    print("Quality control....")
    for vi, var in enumerate(obs_variables):
        ic(var)
        if var == "p":
            rtma_vmin = rtma_range["sp_min"]
            rtma_vmax = rtma_range["sp_max"]
        else:
            rtma_vmin = rtma_range[f"{var}_min"]
            rtma_vmax = rtma_range[f"{var}_max"]

        if var == "t":
            relax_value = 3
        if var == "q":
            relax_value = 0.01
        if var == "u10":
            relax_value = 3
        if var == "v10":
            relax_value = 3
        if var == "p":
            relax_value = 100

        ic(np.nanmin(obs_data[vi]), np.nanmax(obs_data[vi]))
        ic(obs_data[vi].shape)
        obs_data[vi][
            (obs_data[vi] < rtma_vmin - relax_value)
            | (obs_data[vi] > rtma_vmax + relax_value)
        ] = np.nan
        print("After quality control:")
        ic(np.nanmin(obs_data[vi]), np.nanmax(obs_data[vi]))

    # Normalize observations
    obs_data = min_max_norm_ignore_extreme_fill_nan(
        obs_data, ["sta_" + var for var in obs_variables]
    )
    print("Normalization")
    ic(np.nanmin(obs_data[0]), np.nanmax(obs_data[0]))
    ic(np.nanmin(obs_data[1]), np.nanmax(obs_data[1]))
    ic(np.nanmin(obs_data[2]), np.nanmax(obs_data[2]))
    ic(np.nanmin(obs_data[3]), np.nanmax(obs_data[3]))
    ic(np.nanmin(obs_data[4]), np.nanmax(obs_data[4]))

    # %% Satelite
    print("\nReading Satelite....")
    exists_flag, satelite_data, satelite_lat, satelite_lon = (
        read_data_from_satelite_in_time_window(
            satelite_dir=satelite_dir,
            variable_names=satelite_variables,
            obs_time_window=obs_time_window,
            year=analysis_year,
            month=analysis_month,
            day=analysis_day,
            hour=analysis_hour,
            lon_range=lon_range,
            lat_range=lat_range,
        )
    )
    if exists_flag is False:
        return
    ic(satelite_data.shape, np.min(satelite_data), np.max(satelite_data))

    # %% Saving
    ds = xr.Dataset(
        {
            # topography, normalized
            "z": (("lat", "lon"), topo),
            "obs_mask": (("lat", "lon"), obs_mask),
            # %% rtma, normalized
            f"rtma_{rtma_variables[0]}": (
                ("lat", "lon"), rtma_data[0]),
            f"rtma_{rtma_variables[1]}": (
                ("lat", "lon"), rtma_data[1]),
            f"rtma_{rtma_variables[2]}": (
                ("lat", "lon"), rtma_data[2]),
            f"rtma_{rtma_variables[3]}": (
                ("lat", "lon"), rtma_data[3]),
            f"rtma_{rtma_variables[4]}": (
                ("lat", "lon"), rtma_data[4]),
            # %% observation, normalized, 0 means non-station
            f"sta_{obs_variables[0]}": (
                ("obs_time_window", "lat", "lon"), obs_data[0]),
            f"sta_{obs_variables[1]}": (
                ("obs_time_window", "lat", "lon"), obs_data[1]),
            f"sta_{obs_variables[2]}": (
                ("obs_time_window", "lat", "lon"), obs_data[2]),
            f"sta_{obs_variables[3]}": (
                ("obs_time_window", "lat", "lon"), obs_data[3]),
            f"sta_{obs_variables[4]}": (
                ("obs_time_window", "lat", "lon"), obs_data[4]),
            # %% satelite, normalized
            f"{satelite_variables[0]}": (
                ("obs_time_window", "lat_satelite", "lon_satelite"),
                satelite_data[0],
            ),
            f"{satelite_variables[1]}": (
                ("obs_time_window", "lat_satelite", "lon_satelite"),
                satelite_data[1],
            ),
            f"{satelite_variables[2]}": (
                ("obs_time_window", "lat_satelite", "lon_satelite"),
                satelite_data[2],
            ),
            f"{satelite_variables[3]}": (
                ("obs_time_window", "lat_satelite", "lon_satelite"),
                satelite_data[3],
            ),
            # %% HRRR, unnormalized
            f"hrrr_{hrrr_variables[0]}": (("lat", "lon"), hrrr_data[0]),
            f"hrrr_{hrrr_variables[1]}": (("lat", "lon"), hrrr_data[1]),
            f"hrrr_{hrrr_variables[2]}": (("lat", "lon"), hrrr_data[2]),
            f"hrrr_{hrrr_variables[3]}": (("lat", "lon"), hrrr_data[3]),
            f"hrrr_{hrrr_variables[4]}": (("lat", "lon"), hrrr_data[4]),
        },
        coords={
            "valid_time": analysis_time,
            "lat": hrrr_lat,
            "lon": hrrr_lon,
            "lat_satelite": satelite_lat,
            "lon_satelite": satelite_lon,
            "obs_time_window": np.arange(0, obs_time_window),
        },
    )
    print(f"Saving hourly sample to {out_file}")
    ds.to_netcdf(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--index_file", type=str)
    parser.add_argument("--stats_file", type=str)
    parser.add_argument("--obs_dir", type=str)
    parser.add_argument("--satelite_dir", type=str)
    parser.add_argument("--hrrr_dir", type=str)
    parser.add_argument("--hrrr_file", type=str)
    parser.add_argument("--rtma_dir", type=str)
    parser.add_argument("--topography_file", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--hold_out_obs_ratio", type=float)
    # hours before analysis time, interpolate to the analysis time
    parser.add_argument("--obs_time_window", type=int, default=3)
    # Background lead time range
    parser.add_argument("--lead_time", type=int)
    parser.add_argument("--hrrr_variables", nargs="+")
    parser.add_argument("--rtma_variables", nargs="+")
    parser.add_argument("--obs_variables", nargs="+")
    parser.add_argument("--satelite_variables", nargs="+")
    parser.add_argument("--lat_range", nargs="+")
    parser.add_argument("--lon_range", nargs="+")
    parser.add_argument("--overwrite", type=str2bool, default=True)

    args = parser.parse_args()
    obs_dir = args.obs_dir
    hrrr_dir = args.hrrr_dir
    rtma_dir = args.rtma_dir
    topography_file = args.topography_file
    stats_file = args.stats_file
    out_dir = args.out_dir
    satelite_dir = args.satelite_dir

    lon_range = args.lon_range  # [239.975, 287.975]
    lat_range = args.lat_range  # [25.975, 49.975]

    obs_time_window = args.obs_time_window
    lead_time = args.lead_time

    hold_out_obs_ratio = args.hold_out_obs_ratio

    hrrr_variables = args.hrrr_variables
    rtma_variables = args.rtma_variables
    obs_variables = args.obs_variables
    satelite_variables = args.satelite_variables

    overwrite = bool(args.overwrite)

    index_file = args.index_file
    index_df = pd.read_csv(index_file)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df["time_start"] = pd.to_datetime(index_df["time_start"])
    index_df.reset_index(drop=True, inplace=True)
    for index, row in index_df[150:].iterrows():
        print(f"----------------{index}/{len(index_df)}-------------------")
        ic(row)
        analysis_year = row["time_start"].year
        analysis_month = row["time_start"].month
        analysis_day = row["time_start"].day
        analysis_hour = row["time_start"].hour

        generate_samples_hourly(
            analysis_year=analysis_year,
            analysis_month=analysis_month,
            analysis_day=analysis_day,
            analysis_hour=analysis_hour,
            obs_time_window=obs_time_window,
            lead_time=lead_time,
            lon_range=lon_range,
            lat_range=lat_range,
            hold_out_obs_ratio=hold_out_obs_ratio,
            hrrr_variables=hrrr_variables,
            rtma_variables=rtma_variables,
            obs_variables=obs_variables,
            satelite_variables=satelite_variables,
            satelite_dir=satelite_dir,
            rtma_dir=rtma_dir,
            hrrr_dir=hrrr_dir,
            obs_dir=obs_dir,
            topography_file=topography_file,
            stats_file=stats_file,
            out_dir=out_dir,
            overwrite=overwrite,
        )
