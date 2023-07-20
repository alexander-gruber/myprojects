import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm, BoundaryNorm

# from cartopy import ccrs

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from myprojects.io.grids import EASE2

def plot_ease_img(data, tag,
                  llcrnrlat=-60,
                  urcrnrlat=90,
                  llcrnrlon=-180,
                  urcrnrlon=180,
                  cbrange=None,
                  cmap='viridis',
                  norm=False,
                  title='',
                  fontsize=20,
                  log=False,
                  plot_cb=False,
                  print_median=False,
                  plot_label=None):

    grid = EASE2('M36')

    lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)

    img = np.empty(lons.shape, dtype='float32')
    img.fill(None)

    ind_lat = data.loc[:,'ease_row'].values.astype('int')
    ind_lon = data.loc[:,'ease_col'].values.astype('int')

    img[ind_lat,ind_lon] = data.loc[:,tag]
    img_masked = np.ma.masked_invalid(img)

    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    if log:
        im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True, norm=LogNorm())
    else:
        if norm:
            cmap = plt.cm.viridis
            norm = BoundaryNorm(np.arange(norm), cmap.N)
            im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, norm=norm, latlon=True)
        else:
            im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)

    if cbrange is not None:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    else:
        cbrange = [img_masked.min(), img_masked.max()]
    if plot_cb is True:

        ticks = np.arange(cbrange[0],cbrange[1]+0.001, (cbrange[1]-cbrange[0])/4)
        cb = m.colorbar(im, "bottom", size="8%", pad="4%", ticks=ticks)
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize-2)

    plt.title(title,fontsize=fontsize-2)

    if print_median is True:
        x, y = m(-79, 25)
        plt.text(x, y, 'mean = %.2f' % np.ma.mean(img_masked), fontsize=fontsize-2)

    if plot_label:
        x, y = m(-126.5, 25.7)
        plt.text(x, y, plot_label, fontsize=fontsize)

    return im


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



def plot_n_matches():


    res = pd.read_csv(r"H:\work\experiments\temporal_matching_uncertainty\results\ASC_AMS_SMP_AD_12h\result.csv", index_col=0)

    dir_out = Path(r'H:\work\experiments\temporal_matching_uncertainty\plots\n_matches\ASC_AMS_SMP_AD_12h')
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    cols = [f'n_{dt}' for dt in np.arange(24)]
    cols2 = [f'n_{dt}' for dt in np.arange(0,24,2)]

    tmpres = res[cols]
    tmpres.columns = np.arange(24)
    res['n_diff'] = (tmpres.max(axis=1) / tmpres.min(axis=1) - 1) * 100
    res['max_hr'] = tmpres.idxmax(axis=1)

    fontsize = 18

    f = plt.figure(figsize=(25,11))

    for i, col in enumerate(cols2):
        plt.subplot(3,4,i+1)
        im = plot_ease_img(res, col,
                           title=col,
                           cbrange=[200,1400],
                           fontsize=fontsize)

    plot_centered_cbar(f, im, 4, fontsize=fontsize - 2, bottom=0.05, wdth=0.02)

    f.savefig(dir_out / 'n_matches.png', dpi=300, bbox_inches='tight')
    plt.close()

    f = plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_ease_img(res, 'n_diff',
                       title='% difference in matches',
                       plot_cb=True,
                       cbrange=[20,160],
                       fontsize=fontsize)

    plt.subplot(1, 2, 2)
    plot_ease_img(res, 'max_hr',
                       title='reference time with max. matches',
                       plot_cb=True,
                       norm=24,
                       fontsize=fontsize)

    f.savefig(dir_out / 'n_matches_diff.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def plot_correlations():

    mode = 'ASC_AMS_SMP_AD'

    root = Path(r'H:\work\experiments\temporal_matching_uncertainty')

    res = pd.read_csv(root / 'results' / mode / 'result.csv', index_col=0)

    dir_out = root / 'plots' / 'corr' / 'ASC_AMS_SMP_AD'

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    # cols = [f'r_{dt}' for dt in np.arange(24)]
    #
    # tmpres = res[cols]
    # tmpres.columns = np.arange(24)
    # res['n_diff'] = (tmpres.max(axis=1) / tmpres.min(axis=1) - 1) * 100
    # res['max_hr'] = tmpres.idxmax(axis=1)

    fontsize = 18

    mode = ['raw', 'anom_clim', 'anom_st']
    sens = ['ASCAT_AMSR2', 'SMAP_AMSR2', 'SMAP_ASCAT']

    for s in sens:
        for m in mode:

            f = plt.figure(figsize=(25,11))

            cols = [f'r_{m}_{dt}_{s}' for dt in np.arange(0,24,2)]
            for i, col in enumerate(cols):

                plt.subplot(3,4,i+1)
                im = plot_ease_img(res, col,
                                   title=col,
                                   cbrange=[0,1],
                                   fontsize=fontsize)

            plot_centered_cbar(f, im, 4, fontsize=fontsize - 2, bottom=0.05, wdth=0.02)

            f.savefig(dir_out / f'r_{m}_{s}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_correlation_diffs():

    mode = 'ASC_AMS_SMP_AD'

    root = Path(r'H:\work\experiments\temporal_matching_uncertainty')

    res = pd.read_csv(root / 'results' / mode / 'result.csv', index_col=0)

    dir_out = root / 'plots' / 'corr' / mode

    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    fontsize = 18

    mode = ['raw', 'anom_clim', 'anom_st']
    sens = ['ASCAT_AMSR2', 'SMAP_AMSR2', 'SMAP_ASCAT']

    for s in sens:
        for m in mode:

            cols = [f'r_{m}_{dt}_{s}' for dt in np.arange(24)]
            tmpres = res[cols]
            tmpres.columns = np.arange(24)
            res['n_diff'] = tmpres.max(axis=1)- tmpres.min(axis=1)
            res['max_hr'] = tmpres.idxmax(axis=1)

            f = plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plot_ease_img(res, 'n_diff',
                          title='Correlation difference',
                          plot_cb=True,
                          cbrange=[0, 0.3],
                          fontsize=fontsize)

            plt.subplot(1, 2, 2)
            plot_ease_img(res, 'max_hr',
                          title='reference time with max. matches',
                          plot_cb=True,
                          norm=24,
                          fontsize=fontsize)

            f.savefig(dir_out / f'r_diff_{m}_{s}.png', dpi=300, bbox_inches='tight')
            plt.close()


if __name__=='__main__':

    # plot_n_matches()

    # plot_correlations()

    plot_correlation_diffs()

















