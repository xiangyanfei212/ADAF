import os
import glob
import time
import torch
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
import xarray as xr

from math import sqrt
from icecream import ic
from str2bool import str2bool
from collections import OrderedDict
from sklearn.metrics import mean_squared_error

from utils.logging_utils import config_logger, log_to_file
from utils.YParams import YParams
from utils.read_txt import read_lines_from_file

config_logger()


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return x + noise


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint["model_state"].items():
            name = key[7:]
            if name != "ged":
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except ValueError:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def setup(params):

    # device init
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    if params.nettype == "EncDec":
        from models.encdec import EncDec as model
    else:
        raise Exception("not implemented")

    checkpoint_file = params["best_checkpoint_path"]
    logging.info("Loading model checkpoint from {}".format(checkpoint_file))
    model = model(params).to(device)
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    files_paths = glob.glob(params.test_data_path + "/*.nc")
    files_paths.sort()

    return files_paths, inference_times, model


def min_max_norm(data, vmin, vmax):
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

    data = (data - vmin) / (vmax - vmin)
    return data


def min_max_norm_ignore_extreme_fill_nan(data, vmin, vmax):
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

def reverse_norm(params, data, variable_names):

    stats_file = os.path.join(
        params.data_path,
        f"stats_{params.norm_type}.csv")
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

    if params.normalization == "minmax_ignore_extreme":
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

    else:
        raise Exception("not implemented")

    return data


def inference(
    params,
    target_variable,
    test_data_file_paths,
    inference_times,
    hold_out_obs_ratio,
    model,
):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    out_dir = os.path.join(
        params["experiment_dir"],
        f"inference_ensemble_{params.ensemble_num}_hold_{hold_out_obs_ratio}",
    )
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for f, analysis_time_str in zip(test_data_file_paths, inference_times):
            analysis_time = datetime.datetime.strptime(
                analysis_time_str, "%Y-%m-%d_%H")
            logging.info("-----------------------------------------")
            logging.info(f"Analysis time: {analysis_time_str}")
            logging.info(f"Reading {f}")

            out_file = os.path.join(
                out_dir, analysis_time_str + ".nc")

            if not os.path.exists(f):
                logging.info(f"{f} not exists, skip!")
                continue

            data = read_sample_file_and_norm_input(
                params, f, hold_out_obs_ratio)
            (
                inp,  # normed
                inp_sate_norm,  # normed
                inp_hrrr_norm,  # normed
                field_target,  # normed
                hold_out_obs,  # normed
                inp_obs_for_eval,  # normed
                bg_hrrr,  # un-norm
                mask,
                lat,
                lon,
            ) = data

            field_target = reverse_norm(
                params, field_target, params.field_tar_vars)
            # mask the region out of range, after reverse normalization
            field_target = np.where(
                mask, field_target, np.nan
            )  # fill data with nan where mask is True.
            bg_hrrr = np.where(
                mask, bg_hrrr, np.nan
            )  # fill data with nan where mask is True.

            # Reverse normalization
            hold_out_obs = np.where(
                hold_out_obs == 0, np.nan, hold_out_obs
            )  # fill 0 with nan before normalization reversing
            hold_out_obs = reverse_norm(
                params, hold_out_obs, params.inp_obs_vars)

            inp_obs_for_eval = np.where(
                inp_obs_for_eval == 0, np.nan, inp_obs_for_eval
            )  # fill 0 with nan before normalization reversing
            inp_obs_for_eval = reverse_norm(
                params, inp_obs_for_eval, params.inp_obs_vars
            )

            # Unit convert: g/kg -> kg/kg
            q_idx = target_variable.index("q")
            field_target[q_idx] = field_target[q_idx] * 1000
            hold_out_obs[q_idx] = hold_out_obs[q_idx] * 1000
            inp_obs_for_eval[q_idx] = inp_obs_for_eval[q_idx] * 1000
            bg_hrrr[q_idx] = bg_hrrr[q_idx] * 1000

            inp = torch.tensor(inp[np.newaxis, :, :, :]).to(
                device, dtype=torch.float)
            inp_sate_norm = torch.tensor(
                inp_sate_norm[np.newaxis, :, :, :, :]).to(
                    device, dtype=torch.float
            )

            gen_ensembles = []
            for i in range(params.ensemble_num):
                model.eval()
                enable_dropout(model)

                start = time.time()

                if params.nettype == "EncDec":
                    inp_sate_norm = torch.reshape(
                        inp_sate_norm,
                        (1, -1, params.img_size_y, params.img_size_x)
                    )
                    inp = torch.concat((inp, inp_sate_norm), 1)
                    gen = model(inp)
                else:
                    raise Exception("not implemented")

                print(f"inference time: {time.time() - start}")

                if params.learn_residual:
                    gen = np.squeeze(
                        gen.detach().cpu().numpy()) + inp_hrrr_norm

                # reverse normalization
                gen = reverse_norm(params, gen, params.field_tar_vars)

                # for specific humidity, g/kg -> kg/kg
                gen[q_idx] = gen[q_idx] * 1000

                # mask the region out of range with 0,
                # after reverse normalization
                gen = np.where(mask, gen, np.nan)

                gen_ensembles.append(gen)

            gen_ensembles = np.array(gen_ensembles)

            for vi, tar_var in enumerate(target_variable):
                logging.info(f"{tar_var}:")

                # %% compare with hold-out observation
                obs_hold_obs_var = hold_out_obs[vi][
                    ~np.isnan(hold_out_obs[vi])]
                if len(obs_hold_obs_var) == 0:
                    print("No hold out obs, continue!")
                    continue
                bg_hrrr_hold_obs = bg_hrrr[vi][
                    ~np.isnan(hold_out_obs[vi])]
                ai_gen_hold_obs = gen_ensembles[0, vi][
                    ~np.isnan(hold_out_obs[vi])]
                obs_hold_obs_var = np.nan_to_num(
                    obs_hold_obs_var, nan=0)
                bg_hrrr_hold_obs = np.nan_to_num(
                    bg_hrrr_hold_obs, nan=0)
                ai_gen_hold_obs = np.nan_to_num(
                    ai_gen_hold_obs, nan=0)
                rmse_ai_hold_obs = round(
                    sqrt(mean_squared_error(
                        obs_hold_obs_var, ai_gen_hold_obs)), 3
                )
                rmse_bg_hold_obs = round(
                    sqrt(mean_squared_error(
                        obs_hold_obs_var, bg_hrrr_hold_obs)), 3
                )
                logging.info(f"rmse_ai_hold_obs={rmse_ai_hold_obs}")
                logging.info(f"rmse_bg_hold_obs={rmse_bg_hold_obs}")

                # %% compare with input observation
                inp_obs_for_eval_var = inp_obs_for_eval[vi][
                    ~np.isnan(inp_obs_for_eval[vi])
                ]
                bg_hrrr_inp_obs = bg_hrrr[vi][
                    ~np.isnan(inp_obs_for_eval[vi])]
                ai_gen_inp_obs = gen_ensembles[0, vi][
                    ~np.isnan(inp_obs_for_eval[vi])]
                inp_obs_for_eval_var = np.nan_to_num(
                    inp_obs_for_eval_var, nan=0)
                bg_hrrr_inp_obs = np.nan_to_num(
                    bg_hrrr_inp_obs, nan=0)
                ai_gen_inp_obs = np.nan_to_num(
                    ai_gen_inp_obs, nan=0)
                rmse_ai_inp_obs = round(
                    sqrt(mean_squared_error(
                        inp_obs_for_eval_var, ai_gen_inp_obs)), 3
                )
                rmse_bg_inp_obs = round(
                    sqrt(mean_squared_error(
                        inp_obs_for_eval_var, bg_hrrr_inp_obs)), 3
                )
                logging.info(f"rmse_ai_inp_obs={rmse_ai_inp_obs}")
                logging.info(f"rmse_bg_inp_obs={rmse_bg_inp_obs}")

                rmse_ai_field = round(
                    sqrt(
                        mean_squared_error(
                            np.nan_to_num(gen_ensembles[0, vi], nan=0),
                            np.nan_to_num(field_target[vi], nan=0),
                        )
                    ),
                    3,
                )
                logging.info(f"rmse_ai_field={rmse_ai_field}")

                rmse_bg_field = round(
                    sqrt(
                        mean_squared_error(
                            np.nan_to_num(bg_hrrr[vi], nan=0),
                            np.nan_to_num(field_target[vi], nan=0),
                        )
                    ),
                    3,
                )
                logging.info(f"rmse_bg_field={rmse_bg_field}")

                logging.info(
                    "AI generation    :"
                    + f"{round(np.nanmin(gen_ensembles[0,vi]), 3)}"
                    + f"~ {round(np.nanmax(gen_ensembles[0,vi]), 3)}"
                )
                logging.info(
                    "Background (hrrr):"
                    + f"{round(np.nanmin(bg_hrrr[vi]), 3)}"
                    + f"~ {round(np.nanmax(bg_hrrr[vi]), 3)}"
                )
                logging.info(
                    "hold out obs     :"
                    + f"{round(np.nanmin(hold_out_obs[vi]), 3)}"
                    + f"~ {round(np.nanmax(hold_out_obs[vi]), 3)}"
                )
                logging.info(
                    "field_target     :"
                    + f"{round(np.nanmin(field_target[vi]), 3)}"
                    + f"~ {round(np.nanmax(field_target[vi]), 3)}"
                )

            variable_names = [s.split("_")[1] for s in params.field_tar_vars]
            ic(variable_names)
            save_output(
                save_file_path=out_file,
                analysis_time=analysis_time,
                variable_names=variable_names,
                AI_gen_ensembles=gen_ensembles,  # un-normed
                bg_hrrr=bg_hrrr,  # un-normed
                field_target=field_target,  # un-normed
                inp_obs_for_eval=inp_obs_for_eval,  # un-normed
                hold_out_obs=hold_out_obs,  # un-normed
                lon=lon,
                lat=lat,
                mask=mask,
            )


def save_output(
    save_file_path: str,
    variable_names: list,
    AI_gen_ensembles: np.array,
    bg_hrrr: np.array,
    field_target: np.array,
    hold_out_obs: np.array,
    inp_obs_for_eval: np.array,
    lon: np.array,
    lat: np.array,
    mask: np.array,
    analysis_time: str,
):

    ic(
        lon.shape,
        lat.shape,
        AI_gen_ensembles.shape,
        bg_hrrr.shape,
        field_target.shape,
        hold_out_obs.shape,
    )
    ds = xr.Dataset(
        {
            f"ai_gen_{variable_names[0]}": (
                ("ensemble_num", "lat", "lon"),
                AI_gen_ensembles[:, 0],
            ),
            f"ai_gen_{variable_names[1]}": (
                ("ensemble_num", "lat", "lon"),
                AI_gen_ensembles[:, 1],
            ),
            f"ai_gen_{variable_names[2]}": (
                ("ensemble_num", "lat", "lon"),
                AI_gen_ensembles[:, 2],
            ),
            f"ai_gen_{variable_names[3]}": (
                ("ensemble_num", "lat", "lon"),
                AI_gen_ensembles[:, 3],
            ),
            # f"ai_gen_{variable_names[4]}": (
            #     ("ensemble_num", "lat", "lon"),
            #     AI_gen_ensembles[:, 4],
            # ),
            f"hold_out_obs_{variable_names[0]}": (
                ("lat", "lon"), hold_out_obs[0]),
            f"hold_out_obs_{variable_names[1]}": (
                ("lat", "lon"), hold_out_obs[1]),
            f"hold_out_obs_{variable_names[2]}": (
                ("lat", "lon"), hold_out_obs[2]),
            f"hold_out_obs_{variable_names[3]}": (
                ("lat", "lon"), hold_out_obs[3]),
            # f"hold_out_obs_{variable_names[4]}": (
            #   ("lat", "lon"), hold_out_obs[4]),
            f"inp_obs_for_eval_{variable_names[0]}": (
                ("lat", "lon"),
                inp_obs_for_eval[0],
            ),
            f"inp_obs_for_eval_{variable_names[1]}": (
                ("lat", "lon"),
                inp_obs_for_eval[1],
            ),
            f"inp_obs_for_eval_{variable_names[2]}": (
                ("lat", "lon"),
                inp_obs_for_eval[2],
            ),
            f"inp_obs_for_eval_{variable_names[3]}": (
                ("lat", "lon"),
                inp_obs_for_eval[3],
            ),
            f"rtma_{variable_names[0]}": (
                ("lat", "lon"), field_target[0]),
            f"rtma_{variable_names[1]}": (
                ("lat", "lon"), field_target[1]),
            f"rtma_{variable_names[2]}": (
                ("lat", "lon"), field_target[2]),
            f"rtma_{variable_names[3]}": (
                ("lat", "lon"), field_target[3]),
            # f"rtma_{variable_names[4]}": (
            #   ("lat", "lon"), field_target[4]),
            f"bg_hrrr_{variable_names[0]}": (
                ("lat", "lon"), bg_hrrr[0]),
            f"bg_hrrr_{variable_names[1]}": (
                ("lat", "lon"), bg_hrrr[1]),
            f"bg_hrrr_{variable_names[2]}": (
                ("lat", "lon"), bg_hrrr[2]),
            f"bg_hrrr_{variable_names[3]}": (
                ("lat", "lon"), bg_hrrr[3]),
            # f"bg_hrrr_{variable_names[4]}": (
            #   ("lat", "lon"), bg_hrrr[4]),
            "variable_names": (
                ("variable_num"), variable_names),
            "mask": (
                ("lat", "lon"), mask[0]),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "time": analysis_time,
            "time_window": np.arange(0, params.obs_time_window),
            "ensemble_num": np.arange(0, params.ensemble_num),
        },
    )
    logging.info(f"Saving result to {save_file_path}")
    ds.to_netcdf(save_file_path)
    ds.close()


def read_sample_file_and_norm_input(
    params,
    file_path,
    hold_out_obs_ratio=0.2
):

    # %% get statistic
    stats_file = os.path.join(
        params.data_path, f"stats.csv")

    stats = pd.read_csv(stats_file, index_col=0)

    for vi, var in enumerate(params.inp_hrrr_vars):
        if vi == 0:
            inp_hrrr_stats = stats[stats["variable"].isin([var])]
        else:
            inp_hrrr_stats = pd.concat(
                [inp_hrrr_stats, stats[stats["variable"].isin([var])]]
            )

    for vi, var in enumerate(params.inp_obs_vars):
        if vi == 0:
            inp_obs_stats = stats[stats["variable"].isin([var])]
        else:
            inp_obs_stats = pd.concat(
                [inp_obs_stats, stats[stats["variable"].isin([var])]]
            )

    # %% get sample
    ds = xr.open_dataset(file_path, engine="netcdf4")

    lat = np.array(ds.coords["lat"].values)[: params.img_size_y]
    lon = np.array(ds.coords["lon"].values)[: params.img_size_x]

    # background
    inp_hrrr = np.array(ds[params.inp_hrrr_vars].to_array())[
        :, : params.img_size_y, : params.img_size_x
    ]
    inp_hrrr = np.squeeze(inp_hrrr)
    mask = inp_hrrr.copy()
    mask[mask != 0] = 1  # set 1 where out of range
    mask = mask.astype(bool)  # True: out of range

    # baseline for evaluation
    bg_hrrr = inp_hrrr.copy()

    # normalization
    inp_hrrr = min_max_norm_ignore_extreme_fill_nan(
        inp_hrrr, inp_hrrr_stats["min"], inp_hrrr_stats["max"]
    )

    # topography (normed)
    topo = np.array(ds[["z"]].to_array())[
        :, :params.img_size_y, :params.img_size_x]

    # satellite
    inp_sate = np.array [params.inp_satelite_vars].to_array())[
        :, -params.obs_time_window:, :params.img_size_y, :params.img_size_x
    ]

    # Observation (normed)
    obs = np.array(ds[params.inp_obs_vars].to_array())[
        :, -params.obs_time_window:, :params.img_size_y, :params.img_size_x
    ]
    # quality control
    obs[(obs <= -1) | (obs >= 1)] = 0

    if params.hold_out_obs:

        if params.seed != 0:
            np.random.seed(params.seed)
            logging.info(f"Using random seed {params.seed}")

        lat_num = obs[0, 0].shape[0]
        lon_num = obs[0, 0].shape[1]

        # [lat, lon] -> [lat * lon]
        obs_tw_begin = obs[0, 0].reshape(-1)
        # find station's indices
        obs_index = np.where(~np.isnan(obs_tw_begin))[0]

        obs_num = len(obs_index)
        hold_out_num = int(obs_num * hold_out_obs_ratio)
        ic(obs_num, hold_out_num)

        # generate mask randomly
        np.random.shuffle(obs_index)
        hold_out_obs_index = obs_index[:hold_out_num]
        input_obs_index = obs_index[hold_out_num:]
        ic(len(hold_out_obs_index), hold_out_obs_index)
        ic(len(input_obs_index), input_obs_index)

        # Mask (lat, lon), hold_out obs=1, input obs = 0
        obs_mask = np.zeros(obs_tw_begin.shape)
        obs_mask[hold_out_obs_index] = 1
        obs_mask = obs_mask.reshape([lat_num, lon_num])

        inp_obs = obs * (1 - obs_mask)  # observation for input
        hold_out_obs = obs * obs_mask  # observation excluding the input

    inp_obs_for_eval = inp_obs[:, -1]  # -1: analysis time
    print(f"inp_obs_for_eval: {inp_obs_for_eval.shape}")

    inp_obs = inp_obs.reshape((-1, params.img_size_y, params.img_size_x))

    # %% target (normed)
    field_target = np.array(ds[params.field_tar_vars].to_array())[
        :, : params.img_size_y, : params.img_size_x
    ]
    ic(field_target.shape)

    inp = np.concatenate((inp_hrrr, inp_obs, topo), axis=0)

    return (
        inp,  # normed
        inp_sate,  # normed
        inp_hrrr,  # normed
        field_target,  # normed
        hold_out_obs[:, -1],  # normed, -1: analaysis time
        inp_obs_for_eval,  # normed
        bg_hrrr,  # un-normed
        mask,  # out_of_range mask
        lat,
        lon,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--exp_dir", default="", type=str)
    parser.add_argument("--test_data_path", default="", type=str)
    parser.add_argument("--net_config", default="EncDec", type=str)
    parser.add_argument("--hold_out_obs_ratio", type=float, default=0.2)

    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, "config.yaml")

    params = YParams(config_path, args.net_config)
    params["resuming"] = False
    params["seed"] = args.seed
    params["experiment_dir"] = args.exp_dir
    params["test_data_path"] = args.test_data_path
    params["best_checkpoint_path"] = os.path.join(
        params["experiment_dir"], "training_checkpoints", "best_ckpt.tar"
    )

    # set up logging
    log_to_file(
        logger_name=None,
        log_filename=os.path.join(params["experiment_dir"], "inference.log"),
    )
    params.log()

    # get data files and model
    test_data_file_paths, inference_times, model = setup(params)

    target_variable = [var.split("_")[1] for var in params.field_tar_vars]

    inference(
        params,
        target_variable,
        test_data_file_paths,
        inference_times,
        args.hold_out_obs_ratio,
        model,
    )

    logging.info("Done")


