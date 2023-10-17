import numpy as np
import pandas as pd

from netCDF4 import Dataset

import seaborn as sns
sns.set_context('talk', font_scale=0.6)
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import cartopy.feature as cfeature
import cartopy.crs as ccrs


def plot_smos_predictability():

    avg_factor = np.sqrt(10)
    th = [25.4792, 3.8027, 2.2591, 1.6306, 1.2854]
    cl = ['slight', 'moderate', 'severe', 'extreme', 'exceptional', 'unpredictable']

    with Dataset('/Users/ag/Downloads/0-ERA5.swvl1_with_1-ASCAT.sm_with_2-SMOS_L2.Soil_Moisture.nc') as ds:
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
    img_ascat[latidx, lonidx] = np.sqrt(10**(snr_ascat/10)) * avg_factor
    img_smos[latidx, lonidx] = np.sqrt(10**(snr_smos/10)) * avg_factor

    img2_ascat = img_ascat.copy()
    img2_smos = img_ascat.copy()

    for i in np.arange(len(th)-1):
        if i==0:
            img2_ascat[img_ascat>=th[i]] = i
            img2_smos[img2_smos>=th[i]] = i
        else:
            img2_ascat[(img_ascat >= th[i])&(img_ascat < th[i-1])] = i
            img2_smos[(img2_smos >= th[i])&(img2_smos < th[i-1])] = i
    img2_ascat[img_ascat < th[i]] = i+1
    img2_smos[img2_smos < th[i]] = i+1

    fig = plt.figure(figsize=(14, 4))
    fig.suptitle('Minimum required severity for reliable drought detection (monthly average)')
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    # cmap = 'viridis'
    cmap = get_cmap("magma", len(th)+1)
    lim = [0, len(th)+1]
    title1 = 'ASCAT'
    title2 = 'SMOS'


    ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax.set_title(title1)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img2_ascat), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    # cbar = fig.colorbar(img, orientation='vertical', label=cmap_label, ticks=np.arange(len(cl))+0.5)
    # cbar.ax.set_yticklabels(cl)

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_title(title2)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    img = ax.imshow(np.flipud(img2_smos), extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())
    cbar = fig.colorbar(img, orientation='vertical', ticks=np.arange(len(cl))+0.5)
    cbar.ax.set_yticklabels(cl)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    plot_smos_predictability()