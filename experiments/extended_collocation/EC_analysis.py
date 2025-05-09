
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from math import floor
from itertools import repeat, combinations
from multiprocessing import Pool

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.9)
import colorcet as cc

# from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

# from validation_good_practice.ancillary.paths import Paths
# from validation_good_practice.plots import plot_ease_img

from pygeogrids.netcdf import load_grid

from myprojects.io.readers import reader
from pytesmo.metrics import ecol

def run(nprocs=None):

    if not nprocs:
        nprocs = os.cpu_count()-4

    if nprocs == 1:
        EC_test(nprocs, nprocs)
    else:
        p = Pool(nprocs)
        part = np.arange(nprocs) + 1
        parts = repeat(nprocs, nprocs)
        p.starmap(EC_test, zip(part, parts))


def EC_test(part, parts):

    dir_out = Path(r"H:\work\experiments\extended_collocation\results")
    if not dir_out.exists():
        Path.mkdir(dir_out)
    result_file = dir_out / f'result_part{part}.csv'

    if result_file.exists():
        idx = pd.read_csv(result_file, index_col=0).index.values
    else:
        idx = []

    names = ['GLDAS', 'ASCAT', 'AMSR2', 'SMOS', 'SMAP']
    combs = list(combinations(names, 2))
    ecorrs = list(combinations(names[2::], 2))

    ncol = ['n', 'lon', 'lat']
    scols = [f'snr_{n}' for n in names]
    sigcols = [f'sig_{n}' for n in names]
    ercols = [f'err_{n}' for n in names]
    rcols = [f'corr_{c[0]}_{c[1]}' for c in combs]
    ecols = [f'err_corr_{e[0]}_{e[1]}' for e in ecorrs]

    # modes = ['absolute','longterm','shortterm']

    gldas = reader('gldas')
    ascat = reader('ascat')
    amsr2 = reader('amsr2')
    smos = reader('smos')
    smap = reader('smap')

    grid = load_grid(r"D:\data_sets\quarter_subgrid_USA.nc")

    arrcell = grid.activearrcell[grid.activearrlon>-125]
    # cells = np.unique(arrcell)
    cells = [459, 531, 601, 710]

    subs = (np.arange(parts + 1) * len(cells) / parts).astype('int')
    subs[-1] = len(cells)
    start = subs[part - 1]
    end = subs[part] - 1
    cells = cells[start:end]

    arrcell = grid.activearrcell
    for i, cell in enumerate(cells):
        print(f'cell {i+1} / {len(cells)}')

        lons = grid.activearrlon[arrcell==cell]
        lats = grid.activearrlat[arrcell==cell]
        gpis = grid.activegpis[arrcell==cell]

        for j, (gpi, lat, lon) in enumerate(zip(gpis, lats, lons)):
            print(f'gpi {j+1} / {len(gpis)}')

            if gpi in idx:
                continue

            res = pd.DataFrame(columns=ncol + scols + sigcols + ercols + rcols + ecols, index=[gpi], dtype='float')
            # lat, lon = 31.5, -83.55

            tss = [ts.read(lon, lat) for ts in [gldas, ascat, amsr2, smos, smap]]
            df = pd.concat(tss, axis='columns').dropna()
            if len(df.columns) < 5:
                print('Not all data sets available')
                continue

            # for mode in modes:
                # if mode == 'absolute':
                #     ts_ins = ts_insitu.copy()
                #     ts_asc = ts_ascat.copy()
                #     ts_smp = ts_smap.copy()
                #     ts_ol = ts_ol.copy()
                #     ts_da = ts_da.copy()
                # else:
                #     ts_ins = calc_anom(ts_ins.copy(), longterm=(mode=='longterm')).dropna()
                #     ts_asc = calc_anom(ts_asc.copy(), longterm=(mode == 'longterm')).dropna()
                #     ts_smp = calc_anom(ts_smp.copy(), longterm=(mode == 'longterm')).dropna()
                #     ts_ol = calc_anom(ts_ol.copy(), longterm=(mode == 'longterm')).dropna()
                #     ts_da = calc_anom(ts_da.copy(), longterm=(mode == 'longterm')).dropna()

            res.loc[gpi, f'n'] = len(df)
            res.loc[gpi, f'lon'] = lon
            res.loc[gpi, f'lat'] = lat

            corr = df.corr()
            for c in combs:
                res.loc[gpi, f'corr_{c[0]}_{c[1]}'] = corr[c[0]][c[1]]

            try:
                ec_res = ecol(df, correlated=ecorrs)
                for n in names:
                    res.loc[gpi, f'snr_{n}'] = ec_res[f'snr_{n}']
                    res.loc[gpi, f'sig_{n}'] = ec_res[f'sig_{n}']
                    res.loc[gpi, f'err_{n}'] = ec_res[f'err_{n}']
                for c in ecorrs:
                    res.loc[gpi, f'err_corr_{c[0]}_{c[1]}'] = ec_res[f'err_corr_{c[0]}_{c[1]}']
            except:
                print('EC failed')
                pass

            if not result_file.exists():
                res.astype('float').to_csv(result_file, float_format='%0.4f')
            else:
                res.astype('float').to_csv(result_file, float_format='%0.4f', mode='a', header=False)



def plot_centered_cbar(f, im, n_cols, wspace=0.04, hspace=0.025, bottom=0.06, pad=0.03, wdth=0.03, fontsize=12, col_offs=0, fig_ind=None):

    f.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom)

    ctr = n_cols/2 * -1
    if ctr % 1 == 0:
        pos1 = f.axes[int(ctr) - 1 - col_offs].get_position()
        pos2 = f.axes[int(ctr) - col_offs].get_position()
        x1 = (pos1.x0 + pos1.x1) / 2
        x2 = (pos2.x0 + pos2.x1) / 2
    else:
        if fig_ind:
            pos = f.axes[fig_ind].get_position()
        else:
            pos = f.axes[floor(ctr)].get_position()
        x1 = pos.x0
        x2 = pos.x1

    cbar_ax = f.add_axes([x1, pad, x2 - x1, wdth])
    cbar = f.colorbar(im, orientation='horizontal', cax=cbar_ax)
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(fontsize)


def plot_result_maps():

    res_dir = Path(r'H:\work\experiments\extended_collocation\plots')
    fout_n = res_dir / 'n.png'
    fout_snr = res_dir / 'r_true.png'
    fout_err = res_dir / 'err.png'
    fout_ecorr = res_dir / 'err_corr.png'

    plot_n = False
    plot_snr = False
    plot_err = True
    plot_ecorr = False

    res = pd.read_csv(r"H:\work\experiments\extended_collocation\results\result.csv", index_col=0)
    # res = pd.read_csv(r"H:\work\experiments\extended_collocation\results_old\result.csv", index_col=0)

    names = ['ASCAT', 'AMSR2', 'SMOS', 'SMAP']
    ecorrs = list(combinations(names[1::], 2))

    # res = res[res['n'] > 100]
    for i, ec in enumerate(ecorrs):
        res[(res[f'err_corr_{ec[0]}_{ec[1]}'] < -1.) | (res[f'err_corr_{ec[0]}_{ec[1]}'] > 1.)] = np.nan

    grid = load_grid(r"D:\data_sets\quarter_subgrid_USA.nc")
    gpis = grid.activegpis

    lats = np.array([grid.activearrlat[gpis==gpi] for gpi in res.index.values]).flatten()
    lons = np.array([grid.activearrlon[gpis==gpi] for gpi in res.index.values]).flatten()

    arrlat = np.arange(lats.min(), lats.max() + 0.25, 0.25)
    arrlon = np.arange(lons.min(), lons.max() + 0.25, 0.25)
    arrlon, arrlat = np.meshgrid(arrlon, arrlat)

    ilat = ((lats - lats.min()) * 4).astype('int')
    ilon = ((lons - lons.min()) * 4).astype('int')

    extent = (-120, -75, 23, 50)
    crs = ccrs.LambertConformal()

    if plot_n:
        f = plt.figure(figsize=(8, 5))
        ax = f.add_subplot(1, 1, 1, projection=crs)
        img = np.full(arrlon.shape, np.nan)
        img[ilat, ilon] =  res['n'].values
        ax.set_title('Sample size')
        ax.set_extent(extent)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, alpha=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        im = ax.pcolormesh(arrlon, arrlat, img, transform=ccrs.PlateCarree(),
                              cmap='viridis', vmin=100, vmax=700)
        plot_centered_cbar(f, im, 1)

        f.savefig(fout_n, dpi=300, bbox_inches='tight')
        plt.close()

    if plot_snr:
        f = plt.figure(figsize=(14, 9))
        ims = []
        for i, name in enumerate(names):
            ax = f.add_subplot(2, 2, i + 1, projection=crs)
            img = np.full(arrlon.shape, np.nan)
            snr_db = res[f'snr_{name}'].values
            snr = 10**(snr_db/10.)
            R_true = np.sqrt(1. / (1 + snr ** (-1)))
            img[ilat, ilon] = R_true
            ax.set_title(name)
            ax.set_extent(extent)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ims += [ax.pcolormesh(arrlon, arrlat, img, transform=ccrs.PlateCarree(),
                                  cmap='viridis', vmin=0.4, vmax=1)]
        plot_centered_cbar(f, ims[-1], 2)

        f.savefig(fout_snr, dpi=300, bbox_inches='tight')
        plt.close()

    if plot_err:
        f = plt.figure(figsize=(14, 9))
        ims = []
        for i, name in enumerate(names):
            ax = f.add_subplot(2, 2, i + 1, projection=crs)
            img = np.full(arrlon.shape, np.nan)
            beta = res[f'sig_SMAP'].values / res[f'sig_{name}'].values
            img[ilat, ilon] = np.abs(np.sqrt(beta * res[f'err_{name}'].values))
            ax.set_title(name)
            ax.set_extent(extent)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ims += [ax.pcolormesh(arrlon, arrlat, img, transform=ccrs.PlateCarree(),
                                  cmap='viridis', vmin=0, vmax=0.06)]
        plot_centered_cbar(f, ims[-1], 2)

        f.savefig(fout_err, dpi=300, bbox_inches='tight')
        plt.close()

    if plot_ecorr:
        f = plt.figure(figsize=(18, 4))
        ims = []
        for i, ec in enumerate(ecorrs):
            ax = f.add_subplot(1, 3, i + 1, projection=crs)
            img = np.full(arrlon.shape, np.nan)
            img[ilat, ilon] = res[f'err_corr_{ec[0]}_{ec[1]}'].values
            ax.set_title(f'{ec[0]} - {ec[1]}')
            ax.set_extent(extent)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.add_feature(cfeature.LAND, alpha=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
            ims += [ax.pcolormesh(arrlon, arrlat, img, transform=ccrs.PlateCarree(),
                                  cmap=cc.cm.bjy, vmin=-0.8, vmax=0.8)]
        plot_centered_cbar(f, ims[-1], 3, wdth=0.05)

        f.savefig(fout_ecorr, dpi=300, bbox_inches='tight')
        plt.close()

        # plt.show()

    plt.show()

if __name__=='__main__':

    # EC_test(1,1)
    # run(nprocs=6)
    # merge_files(r'H:\work\experiments\extended_collocation\results', pattern='result_*.csv', fname='result.csv', delete=False)
    # merge_files(r'H:\work\experiments\extended_collocation\results\old', pattern='result_*.csv', fname='result.csv', delete=False)

    plot_result_maps()
