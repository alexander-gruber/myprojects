import numpy as np
import pandas as pd

from netCDF4 import Dataset

import seaborn as sns
sns.set_context('talk', font_scale=0.6)
import matplotlib.pyplot as plt

import cartopy.feature as cfeature
import cartopy.crs as ccrs


def plot_smos_predictability():

    thresholds = [25.4792, 3.8027, 2.2591, 1.6306, 1.2854]
    classes = ['slight', 'moderate', 'severe', 'extreme', 'exceptional']

    with Dataset(r"H:\work\experiments\drought_predictability\0-ERA5.swvl1_with_1-ASCAT.sm_with_2-SMOS_L2.Soil_Moisture.nc") as ds:
        lats = ds['lat'][:].data
        lons = ds['lon'][:].data
        snr_ascat = ds['snr_1-ASCAT_between_0-ERA5_and_1-ASCAT_and_2-SMOS_L2'][:].data
        snr_smos = ds['snr_2-SMOS_L2_between_0-ERA5_and_1-ASCAT_and_2-SMOS_L2'][:].data

    latarr = np.arange(lats.min(), lats.max()+0.25, 0.25)
    lonarr = np.arange(lons.min(), lons.max()+0.25, 0.25)

    latidx = ((lats - lats.min())/0.25).astype('int')
    lonidx = ((lons - lons.min())/0.25).astype('int')

    img_ascat = np.full((len(latarr), len(lonarr)), np.nan)
    img_smos = np.full((len(latarr), len(lonarr)), np.nan)

    img_ascat[latidx, lonidx] = np.sqrt(10**(snr_ascat/10))
    img_smos[latidx, lonidx] = np.sqrt(10**(snr_smos/10))

    fig = plt.figure(figsize=(20, 6))
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    cmap = 'viridis'
    lim = [0, 2]
    cmap_label = 'SNR [dB]'
    title1 = 'SNR ASCAT'
    title2 = 'SNR SMOS'

    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_title(title1)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img_ascat), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    fig.colorbar(img, orientation='vertical', label=cmap_label)

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_title(title2)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img_smos), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    fig.colorbar(img, orientation='vertical', label=cmap_label)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    plot_smos_predictability()