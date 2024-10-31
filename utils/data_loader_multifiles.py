import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic
from torch.utils.data import dataloader, dataset
from torch.utils.data.distributed import distributedsampler


def get_data_loader(params, files_pattern, distributed, train):
    dataset = getdataset(params, files_pattern, train)

    if distributed:
        sampler = distributedsampler(dataset, shuffle=train)
    else:
        none

    dataloader = dataloader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=false,  # (sampler is none),
        sampler=sampler if train else none,
        drop_last=true,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class getdataset(dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.train = train
        self.location = location
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        # self.add_noise = params.add_noise if train else false
        self._get_files_stats()

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.nc")
        self.files_paths.sort()
        self.n_samples_total = len(self.files_paths)

        logging.info("getting file stats from {}".format(self.files_paths[0]))
        ds = xr.open_dataset(self.files_paths[0], engine="netcdf4")

        # original image shape (before padding)
        self.org_img_shape_x = ds["hrrr_t"].shape[0]
        self.org_img_shape_y = ds["hrrr_t"].shape[1]

        self.files = [none for _ in range(self.n_samples_total)]

        logging.info("number of samples: {}".format(self.n_samples_total))
        logging.info(
            "found data at path {}. number of examples: {}. \
                original image shape: {} x {} x {}".format(
                self.location,
                self.n_samples_total,
                self.org_img_shape_x,
                self.org_img_shape_y,
                self.n_in_channels,
            )
        )

    def _open_file(self, hour_idx):
        _file = xr.open_dataset(self.files_paths[hour_idx], engine="netcdf4")
        self.files[hour_idx] = _file

    def _min_max_norm_ignore_extreme_fill_nan(self, data, vmin, vmax):
        # ic(vmin.shape, vmax.shape, data.shape)
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

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, hour_idx):
        if self.files[hour_idx] is none:
            self._open_file(hour_idx)

        # %% get statistic value for normalization
        stats_file = os.path.join(
            self.params.data_path, f"stats_{self.params.norm_type}.csv"
        )
        stats = pd.read_csv(stats_file, index_col=0)

        for vi, var in enumerate(self.params.inp_hrrr_vars):
            if vi == 0:
                inp_hrrr_stats = stats[stats["variable"].isin([var])]
            else:
                inp_hrrr_stats = pd.concat(
                    [inp_hrrr_stats, stats[stats["variable"].isin([var])]]
                )

        # %% read data in file
        if len(self.params.inp_hrrr_vars) != 0:
            inp_hrrr = np.array(
                self.files[hour_idx][self.params.inp_hrrr_vars].to_array()
            )[:, : self.params.img_size_y, : self.params.img_size_x]
            inp_hrrr = np.squeeze(inp_hrrr)
            # ic(inp_hrrr.shape)

            field_mask = inp_hrrr.copy()
            field_mask[field_mask != 0] = 1  # set 1 where out of range

            # normalization
            inp_hrrr = self._min_max_norm_ignore_extreme_fill_nan(
                inp_hrrr, inp_hrrr_stats["min"], inp_hrrr_stats["max"]
            )

        if len(self.params.inp_obs_vars) != 0:
            obs = np.array(
                self.files[hour_idx][self.params.inp_obs_vars].to_array())[
                    :,
                    -self.params.obs_time_window:,
                    : self.params.img_size_y,
                    : self.params.img_size_x,
            ]

            # inp_obs not includes nan, 0 means un-observaed location
            # use all observation as target
            obs_tar = obs[:, -1]

            # quality control
            obs_tar[(obs_tar <= -1) | (obs_tar >= 1)] = 0

            # this is for label, which is a combination of obs and analysis
            obs_tar_mask = obs_tar.copy()
            # 1 means observed, 0 means un-observed
            obs_tar_mask[obs_tar_mask != 0] = 1
            # print(f'obs_tar: {obs_tar.shape}')

            # print(f'inp_obs: {inp_obs.shape}')
            if self.params.hold_out_obs:
                # [lat, lon]
                # obs_mask = np.array(self.files[hour_idx]["obs_mask"])
                # # print(f'obs_mask: {obs_mask.shape}')
                # inp_obs = inp_obs * (1 - obs_mask)

                if self.params.obs_mask_seed != 0:
                    np.random.seed(self.params.obs_mask_seed)
                    logging.info(
                        f"using random seed {self.params.obs_mask_seed}")

                lat_num = obs[0, 0].shape[0]
                lon_num = obs[0, 0].shape[1]

                # [lat, lon] -> [lat * lon]
                obs_tw_begin = obs[0, 0].reshape(-1)
                obs_index = np.where(~np.isnan(obs_tw_begin))[
                    0
                ]  # find station's indices

                obs_num = len(obs_index)
                hold_out_num = int(obs_num * self.params.hold_out_obs_ratio)
                ic(obs_num, hold_out_num)

                np.random.shuffle(obs_index)  # generate mask randomly
                hold_out_obs_index = obs_index[:hold_out_num]
                # input_obs_index = obs_index[hold_out_num:]
                # ic(len(hold_out_obs_index), hold_out_obs_index)
                # ic(len(input_obs_index), input_obs_index)

                # mask (lat, lon), hold_out obs=1, input obs = 0
                obs_mask = np.zeros(obs_tw_begin.shape)
                obs_mask[hold_out_obs_index] = 1
                obs_mask = obs_mask.reshape([lat_num, lon_num])

                # observation for input
                inp_obs = obs * (1 - obs_mask)
                # observation excluding the input
                # hold_out_obs = obs * obs_mask

                # ic(inp_obs.shape)
                inp_obs = inp_obs.reshape(
                    (-1, self.params.img_size_y, self.params.img_size_x)
                )

        if len(self.params.inp_satelite_vars) != 0:
            inp_sate = np.array(
                self.files[hour_idx][self.params.inp_satelite_vars].to_array()
            )[
                :,
                -self.params.obs_time_window:,
                : self.params.img_size_y,
                : self.params.img_size_x,
            ]

        lon = np.array(self.files[hour_idx].coords["lon"].values)[
            : self.params.img_size_x
        ]
        lat = np.array(self.files[hour_idx].coords["lat"].values)[
            : self.params.img_size_y
        ]
        topo = np.array(self.files[hour_idx][["z"]].to_array())[
            :, : self.params.img_size_y, : self.params.img_size_x
        ]
        field_tar = np.array(
            self.files[hour_idx][self.params.field_tar_vars].to_array()
        )[:, : self.params.img_size_y, : self.params.img_size_x]

        # a combination of observation and target analysis field
        # use observed value to replace the analysis value
        # at observed locations
        field_obs_tar = field_tar.copy()
        # use 0 replace the value at observed location
        field_obs_tar[obs_tar_mask == 1] = 0
        field_obs_tar += obs_tar

        # norm(field_tar) - norm(inp_hrrr)
        # norm(obs_tar) - norm(inp_hrrr)
        # norm(field_obs_tar) - norm(inp_hrrr)
        if self.params.learn_residual:
            field_tar = field_tar - inp_hrrr
            obs_tar = obs_tar - inp_hrrr
            field_obs_tar = field_obs_tar - inp_hrrr

        inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0)

        if len(self.params.inp_satelite_vars) != 0:
            inp_sate = inp_sate.reshape(
                (-1, self.params.img_size_y, self.params.img_size_x)
            )
            inp = np.concatenate((inp, inp_sate))

            return (
                inp,
                field_tar,
                obs_tar,
                field_obs_tar,
                inp_hrrr,
                lat,
                lon,
                field_mask,
                obs_tar_mask,
            )
        else:
            return (
                inp,
                field_tar,
                obs_tar,
                field_obs_tar,
                inp_hrrr,
                lat,
                lon,
                field_mask,
                obs_tar_mask,
            )

