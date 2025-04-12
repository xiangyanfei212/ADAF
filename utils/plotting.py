import os
import numpy as np
import xarray as xr
from icecream import ic

import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# plt.rc('font', family='Times New Roman')
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
# fontdict = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 18}


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except NameError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1] * amount, c[2])


def plot_regression_scatter(
    # data: pd.DataFrame,
    # x_col: str,
    # y_col: str,
    # save_path: str
):

    # make the data
    np.random.seed(3)
    x1 = 4 + np.random.normal(0, 2, 1000)
    y1 = 4 + np.random.normal(0, 2, len(x1))

    x2 = 3 + np.random.normal(0, 2, 1000)
    y2 = 3 + np.random.normal(0, 2, len(x2))

    # Init a figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x1, y1, s=30, alpha=0.7, c="red")
    ax.scatter(x2, y2, s=30, alpha=0.7, c="green")

    # Fit linear regression via least squares with numpy.polyfit
    # It returns an slope (b) and intercept (a)
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    b1, a1 = np.polyfit(x1, y1, deg=1)
    b2, a2 = np.polyfit(x2, y2, deg=1)

    # Create sequence of 100 numbers from 0 to 100
    xseq = np.linspace(0, 10, num=100)

    # Plot regression line
    ax.plot(xseq, a1 + b1 * xseq, color="k", lw=2.5)
    ax.plot(xseq, a2 + b2 * xseq, color="k", lw=2.5)

    ax.set(
        xlim=(0, 8),
        xticks=np.arange(1, 8),
        ylim=(0, 8),
        yticks=np.arange(1, 8)
    )

    # print(f"saving to {save_path}")
    plt.savefig("temp.png")

    plt.close()
    return


def plot_boxplot(
    rmse_df,
    group_names,
    variable_name,
    save_path_suffix,
    font_size=10
):

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    if variable_name == "t":
        title = "T2M"
        units = r"$^{\circ} C$"
    elif variable_name == "q":
        title = "Q"
        units = "$kg/kg$"
    elif variable_name == "u10":
        title = "U10"
        units = "$m/s$"
    elif variable_name == "v10":
        title = "V10"
        units = "$m/s$"
    elif variable_name == "sp":
        title = "SP"
        units = "$hPa$"
    else:
        raise ValueError(
            f"Unexpected variable_name: {variable_name}")

    # Init a figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))

    grouped = rmse_df.groupby("category")["value"]

    # Create the plot with different colors for each group
    ax.set_title(title)
    boxplot = ax.boxplot(
        x=[group.values for name, group in grouped],
        labels=grouped.groups.keys(),
        patch_artist=True,
        medianprops={"color": "black"},
    )
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    # Define colors for each group
    colors = ["#EA0217", "#1079B9", "#FF7500"]

    # Assign colors to each box in the boxplot
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color)

    # Add the p value and the t
    # p_value_text = f'p-value: {p_value}'
    # ax.text(0.7, 50, p_value_text, weight='bold')
    # f_value_text = f'F-value: {F_statistic}'
    # ax.text(0.7, 45, f_value_text, weight='bold')

    # Add a title and axis label
    ax.set_title(title, fontsize=font_size)
    ax.set_ylabel(f"RMSE ({units})", fontsize=font_size)

    # Display it
    # plt.show()

    # Save it
    plt.savefig(
        save_path_suffix + ".png",
        bbox_inches="tight",
        dpi=600,
        pad_inches=0.05
    )
    print(f"save figure to {save_path_suffix + '.png'}")
    plt.savefig(
        save_path_suffix + ".pdf",
        bbox_inches="tight",
        dpi=600,
        pad_inches=0.05
    )
    print(f"save figure to {save_path_suffix + '.pdf'}")

    # Close it
    plt.close()


def plot_spatial_dist_metrics(metrics_spatial_file, metrics_var, save_path):
    ds = xr.open_dataset(metrics_spatial_file)
    lon = ds["lon"].values
    lat = ds["lat"].values

    metrics_value = ds[metrics_var].values.T
    ic(
        metrics_value.shape,
        np.nanmin(metrics_value),
        np.nanmax(metrics_value),
        np.nanmean(metrics_value),
    )
    # metrics_value = np.nan_to_num(metrics_value, 0)

    vmin = np.nanmin(metrics_value)
    vmax = np.nanmax(metrics_value)
    # levels = np.arange(vmin, vmax + 1, 1)
    ic(vmin, vmax)

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    ic(lon_2d.shape, lat_2d.shape)
    leftlon, rightlon, lowerlat, upperlat = (
        np.min(lon_2d),
        np.max(lon_2d),
        np.min(lat_2d),
        np.max(lat_2d),
    )
    extent = [leftlon, rightlon, lowerlat, upperlat]

    fig = plt.figure(figsize=(12, 8), dpi=300)
    crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  # for colorbar

    ax1 = fig.add_subplot(111, projection=proj)
    ax1.set_extent(extent, crs=crs)

    ax1.add_feature(cfeature.COASTLINE, edgecolor="grey")
    ax1.add_feature(cfeature.BORDERS, edgecolor="grey")
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
    )
    ax1.add_feature(states_provinces, edgecolor="gray")

    c1 = ax1.contourf(
        lon_2d,
        lat_2d,
        metrics_value,
        # zorder=0,
        # levels=levels,
        # extend="both",
        transform=crs,
        norm=norm,
        cmap="jet",
    )

    proj0 = ccrs.PlateCarree()
    ax1.set_xticks(np.arange(extent[0], extent[1] + 1, 8), crs=proj0)
    ax1.set_yticks(np.arange(extent[-2], extent[-1] + 1, 4), crs=proj0)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=False))
    ax1.yaxis.set_major_formatter(
        LatitudeFormatter())
    plt.tick_params(labelsize=15)

    # %% colorbar
    cax = fig.add_axes(
        [
            ax1.get_position().x1 + 0.01,
            ax1.get_position().y0,
            0.02,
            ax1.get_position().y1 - ax1.get_position().y0,
        ]
    )
    cb = fig.colorbar(c1, cax=cax)
    # cb.ax.set_ylabel("Temperature", rotation=0, fontdict=fontdict)
    cb.ax.tick_params(labelsize=11, direction="out")

    # plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600, pad_inches=0.05)

    print(f"Save figure to {save_path}")
    plt.close()


def plot_2d_map(
    data_2d: np.array,
    save_path: str,
    lon: list,
    lat: list,
    vmin: float,
    vmax: float,
):

    # vmin = np.nanmin(data_2d)
    # vmax = np.nanmax(data_2d)
    # ic(vmin, vmax)

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    ic(lon_2d.shape, lat_2d.shape)
    leftlon, rightlon, lowerlat, upperlat = (
        np.min(lon_2d),
        np.max(lon_2d),
        np.min(lat_2d),
        np.max(lat_2d),
    )
    extent = [leftlon, rightlon, lowerlat, upperlat]

    fig = plt.figure(figsize=(12, 8), dpi=300)
    crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)

    # for colorbar
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    ax1 = fig.add_subplot(111, projection=proj)
    ax1.set_extent(extent, crs=crs)

    ax1.add_feature(cfeature.COASTLINE, edgecolor="grey")
    ax1.add_feature(cfeature.BORDERS, edgecolor="grey")
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
    )
    ax1.add_feature(states_provinces, edgecolor="gray")

    c1 = ax1.contourf(
        lon_2d,
        lat_2d,
        data_2d,
        # zorder=0,
        # levels=levels,
        # extend="both",
        transform=crs,
        norm=norm,
        cmap="jet",
    )

    proj0 = ccrs.PlateCarree()
    ax1.set_xticks(
        np.arange(extent[0], extent[1] + 1, 8), crs=proj0)
    ax1.set_yticks(
        np.arange(extent[-2], extent[-1] + 1, 4), crs=proj0)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=False))
    ax1.yaxis.set_major_formatter(
        LatitudeFormatter())
    plt.tick_params(labelsize=15)

    # %% colorbar
    cax = fig.add_axes(
        [
            ax1.get_position().x1 + 0.01,
            ax1.get_position().y0,
            0.02,
            ax1.get_position().y1 - ax1.get_position().y0,
        ]
    )
    cb = fig.colorbar(c1, cax=cax)
    # cb.ax.set_ylabel("Temperature", rotation=0, fontdict=fontdict)
    cb.ax.tick_params(labelsize=11, direction="out")

    # plt.suptitle(title)
    # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=600, pad_inches=0.05)

    print(f"Save figure to {save_path}")
    plt.close()





if __name__ == "__main__":

    # sample_size = 100

    # groupA = np.random.normal(10, 10, sample_size)
    # groupB = np.random.normal(70, 10, sample_size)
    # groupC = np.random.normal(40, 10, sample_size)

    # category = ['ADAF']*sample_size + \
    #  ['HRRR']*sample_size + \
    #  ['RTMA']*sample_size

    # df = pd.DataFrame({
    #           'value': np.concatenate([groupA, groupB, groupC]),
    #           'category': category})

    # # Group our dataset with our 'Group' variable
    # grouped = df.groupby('category')['value']

    # plot_boxplot(df, ['ADAF', 'HRRR', 'RTMA'], 'temp.png')

    # plot_regression_scatter()

    # metrics_var = 'ai_rtma_bias'
    # save_path = 'tmp.png'
    # plot_spatial_dist_metrics(
    #   metrics_spatial_file,
    #   metrics_var,
    #   save_path)

    ds = xr.open_dataset(
        os.path.join(
            "weather-blob",
            "users",
            "v-yanfei",
            "WeatherForecasting",
            "dailyforecast",
            "AI_data_assimilation" "exp_us_samples_v9_tqu10v10sp",
            "SwinIR",
            "20240605-150646",
            "inference_ensemble_1",
            "2023-09-27_12.nc",
        )
    )
    lon = ds["lon"].values
    lat = ds["lat"].values
    data_2d = ds["AI_gen_ensembles"].values
    save_path = "./temp.png"

    print(data_2d.shape)
    vmin = np.min(data_2d)
    vmax = np.max(data_2d)

    plot_2d_map(
        data_2d=data_2d,
        save_path=save_path,
        lon=lon,
        lat=lat,
        vmin=vmin,
        vmax=vmax,
    )
