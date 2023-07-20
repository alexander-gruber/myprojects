import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from scipy.ndimage import gaussian_filter
import scipy.optimize as optimization
from itertools import repeat

from myprojects.io.gen_syn_data import generate_soil_moisture, generate_error
from myprojects.io.insitu import ISMN_io
from myprojects.io.smap import SMAP_io
from myprojects.io.ascat import HSAF_io
from myprojects.io.amsr2 import AMSR2_io

from multiprocessing import Pool

from myprojects.timeseries import calc_anom
from myprojects.functions import merge_files

from validation_good_practice.ancillary.metrics import TCA_calc
from pytesmo.temporal_matching import df_match, combined_temporal_collocation


from datetime import datetime as dt


class interface(object):

    def __init__(self):
        self.io_smp = SMAP_io()
        self.io_asc = HSAF_io()
        self.io_lprm = AMSR2_io()

        self.lut = pd.read_csv("D:\data_sets\lut_smap_ascat_amsr2.csv", index_col=0)

        self.names =['SMAP', 'ASCAT', 'AMSR2']
        self.ios = [self.io_smp, self.io_asc, self.io_lprm]

    def read(self, lat, lon, resample_hours=False):

        dss = []
        for name, io in zip(self.names, self.ios):

            if name == 'SMAP':
                tmp_ds = io.read(lat, lon)['soil_moisture']
            else:
                tmp_ds = io.read(lat, lon)

            if resample_hours:
                tmp_ds = tmp_ds['2015-01-01':'2022-01-01'].resample('1h').mean().dropna()
            else:
                tmp_ds = tmp_ds['2015-01-01':'2022-01-01'].dropna()
            tmp_ds.name = 'raw'
            tmp_ds = pd.DataFrame(tmp_ds)
            # tmp_ds['clim'] = calc_anom(tmp_ds['raw'], mode='climatological', return_clim=True)
            tmp_ds['anom_lt'] = calc_anom(tmp_ds['raw'], mode='longterm')
            tmp_ds['anom_clim'] = calc_anom(tmp_ds['raw'], mode='climatological')
            tmp_ds['anom_st'] = calc_anom(tmp_ds['raw'], mode='shortterm')

            dss += [tmp_ds]

        return dss


    def iter_gps(self, resample_hours=False):

        for _, info in self.lut.iterrows():

            dss = []
            for name, io in zip(self.names, self.ios):

                if name == 'SMAP':
                    tmp_ds = io.read(int(info.ease_row), int(info.ease_col), rowcol=True)['soil_moisture']
                elif name == 'ASCAT':
                    try:
                        tmp_ds = io.read(int(info.gpi_asc))
                        if tmp_ds is None:
                            tmp_ds = pd.Series(dtype='float')
                    except:
                        tmp_ds = pd.Series(dtype='float')
                else:
                    tmp_ds = io.read(int(info.gpi_ams))

                if len(tmp_ds) > 0:
                    if resample_hours:
                        tmp_ds = tmp_ds['2015-01-01':'2022-01-01'].resample('1h').mean().dropna()
                    else:
                        tmp_ds = tmp_ds['2015-01-01':'2022-01-01'].dropna()
                    tmp_ds.name = 'raw'
                    tmp_ds = pd.DataFrame(tmp_ds)
                    tmp_ds['anom_clim'] = calc_anom(tmp_ds['raw'], mode='climatological')
                    tmp_ds['anom_lt'] = calc_anom(tmp_ds['raw'], mode='longterm')
                    tmp_ds['anom_st'] = calc_anom(tmp_ds['raw'], mode='shortterm')

                dss += [tmp_ds]

            yield dss, info


def r(a, b):

    return pd.DataFrame({0:a,1:b}).corr()[0][1]


def calc_rmsd(df):

    rmsd = lambda a, b: np.sqrt(np.nanmean((a-b)**2))
    tmp_df = df.dropna()
    cols = df.columns.values
    pairs = [(0,1),(0,2),(1,2)]
    res = [rmsd(tmp_df[cols[p[0]]],tmp_df[cols[p[1]]]) for p in pairs]
    return res

def estimate_tau(in_df, n_lags=24*90):
    """
    Estimate characteristic time lengths for pd.DataFrame columns by fitting an exponential auto-correlation function.

    Parameters
    ----------
    in_df : pd.DataFrame
        Input data frame
    n_lags : maximum allowed lag size [days] to be considered

    """

    # Approx. daily time steps are assumed here. For LSMs and in situ data, sub-daily (e.g. 3-hourly) predictions are
    # disregarded, which increases speed but should not affect the estimate of tau much if time series are long enough.
    df = in_df.copy()
    n_cols = len(df.columns)

    # calculate auto-correlation coefficients for different lags [days]
    rho = np.full((n_cols,n_lags), np.nan)
    for lag in np.arange(n_lags):
        for i,col in enumerate(df):
            Ser_l = df[col].copy()
            Ser_l.index += pd.Timedelta(lag, 'h')
            rho[i,lag] = df[col].corr(Ser_l)


    # Fit exponential function to auto-correlations and estimate tau
    tau = np.full(n_cols, np.nan)
    for i in np.arange(n_cols):
        try:
            ind = np.where(~np.isnan(rho[i,:]))[0]
            if len(ind) > 10:
                popt = optimization.curve_fit(lambda x, a: np.exp(a * x), np.arange(n_lags)[ind], rho[i,ind],
                                              bounds = [-1., -1. / n_lags])[0]
                tau[i] = np.log(np.exp(-1.)) / popt
        except:
            # If fit doesn't converge, fall back to the lag where calculated auto-correlation actually drops below 1/e
            ind = np.where(rho[i,:] < np.exp(-1))[0]
            tau[i] = ind[0] if (len(ind) > 0) else n_lags # maximum = # calculated lags


    # plt.plot(rho[0,:])
    # plt.axhline(np.exp(1)**-1, color='k', linewidth=1, linestyle='--')
    # plt.axvline(tau[0], color='k', linewidth=1, linestyle='--')
    # plt.show()

    return tau

def perturb_data(sm):

    size = len(sm)

    err1 = generate_error(size=size, var=np.nanvar(sm) / 2.)
    err2 = generate_error(size=size, var=np.nanvar(sm) / 2.)
    err3 = generate_error(size=size, var=np.nanvar(sm) / 2.)

    x1 = sm + err1
    x2 = sm + err2
    x3 = sm + err3

    # x1 = sm
    # x2 = sm
    # x3 = sm

    return x1, x2, x3

def gen_data():

    size = 2500
    gamma = 0.96
    scale = 15

    sm, p = generate_soil_moisture(gamma=gamma, scale=scale, size=size)

    x1, x2, x3 = perturb_data(sm)

    return sm, x1, x2, x3


def check_syn_ts():

    nmin = -144
    nmax = -nmin

    dt2 = 100 - nmin
    dt3 = 100 - nmin

    network = 'RSMN'
    station = 'Bacles'
    io = ISMN_io()
    surface_depth = 0.05
    ts = io.read(network, station, surface_only=True, surface_depth=surface_depth).dropna().resample('1h').mean()
    ts.index.name = None
    ts.columns = ['raw']
    ts['clim'] = calc_anom(ts['raw'], mode='climatological', return_clim=True)
    ts['anom_clim'] = calc_anom(ts['raw'], mode='climatological')
    ts['anom_lt'] = calc_anom(ts['raw'], mode='longterm')
    ts['anom_st'] = calc_anom(ts['raw'], mode='shortterm')

    sm = ts['anom_st'].values
    x1, x2, x3 = perturb_data(sm)

    print(r(sm, x1))
    print(r(sm, x2))
    print(r(sm, x3))

    ref = x1[-nmin:len(x1) - nmax]
    nref = len(ref)

    tmp2 = x2[dt2:dt2 + nref]
    tmp3 = x3[dt3:dt3 + nref]
    df = pd.DataFrame({0: ref, 1: tmp2, 2: tmp3})

    print('----------')

    corr = df.corr() ** 2
    print(corr[0][1])
    print(corr[0][2])
    print(corr[1][2])

    tcr, err, beta = TCA_calc(df)
    err/=beta

    print('----------')

    print(abs(tcr[0]))
    print(abs(tcr[1]))
    print(abs(tcr[2]))

    print(abs(err[0]))
    print(abs(err[1]))
    print(abs(err[2]))

    # df.iloc[0:1000,:].plot()
    # plt.tight_layout()
    # plt.show()

def synthetic_experiment():

    # plot_rs = True
    # plot_ts = False

    plot_rs = True
    plot_ts = False

    smooth = False

    if plot_rs:

        nmin = -40
        nmax = -nmin
        stepsize = 2

        dts2 = np.arange(nmin, nmax, stepsize)
        dts3 = np.arange(nmin, nmax, stepsize)
        dtsx, dtsy = np.meshgrid(dts2,dts3)

        reps = 5

        shp = [len(dts2), len(dts3), reps]
        r12 = np.full(shp, np.nan)
        r13 = np.full(shp, np.nan)
        r23 = np.full(shp, np.nan)

        r1 = np.full(shp, np.nan)
        r2 = np.full(shp, np.nan)
        r3 = np.full(shp, np.nan)

        r1_t = np.full(reps, np.nan)
        r2_t = np.full(reps, np.nan)
        r3_t = np.full(reps, np.nan)

        for rep in np.arange(reps):

            cnt = 0
            for ix, dt2 in enumerate(dts2-nmin):
                for iy, dt3 in enumerate(dts3-nmin):
                    cnt += 1
                    print(f'{cnt} / {len(dts2)*len(dts3)} ({rep+1})')

                    sm, x1, x2, x3 = gen_data()

                    r1_t[rep] = r(sm, x1)
                    r2_t[rep] = r(sm, x2)
                    r3_t[rep] = r(sm, x3)

                    ref = x1[-nmin:len(x1) - nmax]
                    nref = len(ref)

                    tmp2 = x2[dt2:dt2+nref]
                    tmp3 = x3[dt3:dt3+nref]
                    df = pd.DataFrame({0:ref, 1:tmp2, 2:tmp3})

                    corr = df.corr()**2
                    r12[iy, ix, rep] = corr[0][1]
                    r13[iy, ix, rep] = corr[0][2]
                    r23[iy, ix, rep] = corr[1][2]

                    tcr, _, _ = TCA_calc(df)

                    # if len(np.where(np.isnan(tcr))[0]) > 0:
                    #     print('nan detected')

                    r1[iy, ix, rep] = abs(tcr[0])
                    r2[iy, ix, rep] = abs(tcr[1])
                    r3[iy, ix, rep] = abs(tcr[2])

        r1_t = r1_t.mean()
        r2_t = r2_t.mean()
        r3_t = r3_t.mean()

        # if np.where(np.isnan(r12))[0] > 0:
        #     print('nan detected')

        if smooth:
            sig = 0.8
            t = 4
            r12 = gaussian_filter(r12.mean(axis=2), sigma=sig, truncate=t)
            r13 = gaussian_filter(r13.mean(axis=2), sigma=sig, truncate=t)
            r23 = gaussian_filter(r23.mean(axis=2), sigma=sig, truncate=t)
            r1 = gaussian_filter(r1.mean(axis=2), sigma=sig, truncate=t)
            r2 = gaussian_filter(r2.mean(axis=2), sigma=sig, truncate=t)
            r3 = gaussian_filter(r3.mean(axis=2), sigma=sig, truncate=t)
        else:
            r12 = r12.mean(axis=2)
            r13 = r13.mean(axis=2)
            r23 = r23.mean(axis=2)
            r1 = r1.mean(axis=2)
            r2 = r2.mean(axis=2)
            r3 = r3.mean(axis=2)


        f, ax = plt.subplots(2, 3, figsize=(15,10), sharey='all', sharex='all')

        vmin = 0.007
        vmax = 0.7

        ax[0, 0].pcolormesh(dtsx, dtsy, r12, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax[0, 0].set_title('R(x1, x2)')
        # ax[0, 0].set_xlabel('dt (x2)')
        ax[0, 0].set_ylabel('dt (x3)')

        ax[0, 1].pcolormesh(dtsx, dtsy, r13, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax[0, 1].set_title('R(x1, x3)')
        # ax[0, 1].set_xlabel('dt (x2)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r23, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax[0, 2].set_title('R(x2, x3)')
        # ax[0, 2].set_xlabel('dt (x2)')

        # cbax = f.add_axes([0.8, 0.1, 0.1, 0.8])
        cb = f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0
        vmax = 0.8

        # ax[1, 0].pcolormesh(dtsx, dtsy, r1, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax[1, 0].pcolormesh(dtsx, dtsy, r1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'R(x1) / true = {r1_t:.2f}')
        ax[1, 0].set_xlabel('dt (x2)')
        ax[1, 0].set_ylabel('dt (x3)')

        # ax[1, 1].pcolormesh(dtsx, dtsy, r2, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        ax[1, 1].pcolormesh(dtsx, dtsy, r2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'R(x2) / true = {r2_t:.2f}')
        ax[1, 1].set_xlabel('dt (x2)')

        # im = ax[1, 2].pcolormesh(dtsx, dtsy, r3, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
        im = ax[1, 2].pcolormesh(dtsx, dtsy, r3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'R(x3) / true = {r3_t:.2f}')
        ax[1, 2].set_xlabel('dt (x2)')

        # cbax = f.add_axes([0.8, 0.1, 0.1, 0.8])
        cb = f.colorbar(im, ax=ax[1, 2], orientation='vertical')

    if plot_ts:
        sm, x1, x2, x3 = gen_data()

        dt = 20
        ref = sm[dt:len(x1) - dt]
        nref = len(ref)

        dts2 = [0, 2*dt, 0, 2*dt]
        dts3 = [0, 0, 2*dt, 2*dt]

        _, ax = plt.subplots(4, 1, figsize=(15, 10), sharex='all')

        for i, (dt2, dt3) in enumerate(zip(dts2, dts3)):

            tmp2 = sm[dt2:dt2 + nref]
            tmp3 = sm[dt3:dt3 + nref]

            df = pd.DataFrame({'x1(t0)':ref,
                               f'x2(t{dt2-dt})':tmp2,
                               f'x3(t{dt3-dt})':tmp3}).dropna()

            df.plot(ax=ax[i])
            # ax[i].plot(sm, color='g', linestyle='--')

    plt.tight_layout()
    plt.show()

def satellite_experiment_rel():

    network = 'RSMN'
    station = 'Bacles'

    # network = 'SMOSMANIA'
    # station = 'Sabres'

    # network = 'AMMA-CATCH'
    # station = 'Belefoungou-Mid'

    io_ins = ISMN_io()
    io_smp = SMAP_io()
    io_asc = HSAF_io()

    lat = io_ins.list[(io_ins.list['network']==network) & (io_ins.list['station']==station)]['latitude'].values[0]
    lon = io_ins.list[(io_ins.list['network'] == network) & (io_ins.list['station'] == station)]['longitude'].values[0]

    ts_ins = io_ins.read(network, station, surface_only=True, surface_depth=0.05).dropna().resample('1h').mean()
    ts_ins.index.name = None
    ts_ins.columns = ['raw']
    ts_ins['clim'] = calc_anom(ts_ins['raw'], mode='climatological', return_clim=True)
    ts_ins['anom_clim'] = calc_anom(ts_ins['raw'], mode='climatological')
    ts_ins['anom_lt'] = calc_anom(ts_ins['raw'], mode='longterm')
    ts_ins['anom_st'] = calc_anom(ts_ins['raw'], mode='shortterm')

    ts_smp = io_smp.read(lat,lon)['soil_moisture']
    ts_smp = ts_smp.resample('1h').mean().dropna()
    ts_smp.name = 'raw'
    ts_smp = pd.DataFrame(ts_smp)
    ts_smp['clim'] = calc_anom(ts_smp['raw'], mode='climatological', return_clim=True)
    ts_smp['anom_clim'] = calc_anom(ts_smp['raw'], mode='climatological')
    ts_smp['anom_lt'] = calc_anom(ts_smp['raw'], mode='longterm')
    ts_smp['anom_st'] = calc_anom(ts_smp['raw'], mode='shortterm')

    ts_asc = io_asc.read(lat, lon)
    ts_asc = ts_asc.resample('1h').mean().dropna()
    ts_asc.name = 'raw'
    ts_asc = pd.DataFrame(ts_asc)
    ts_asc['clim'] = calc_anom(ts_asc['raw'], mode='climatological', return_clim=True)
    ts_asc['anom_clim'] = calc_anom(ts_asc['raw'], mode='climatological')
    ts_asc['anom_lt'] = calc_anom(ts_asc['raw'], mode='longterm')
    ts_asc['anom_st'] = calc_anom(ts_asc['raw'], mode='shortterm')

    modes = ['raw', 'anom_clim', 'anom_lt', 'anom_st']

    dts = np.arange(-24,25)

    rs_smp = pd.DataFrame(columns=modes, index=dts)
    rs_asc = pd.DataFrame(columns=modes, index=dts)

    idx = ts_ins.index
    for dt in dts:
        ts_ins.index = idx + pd.to_timedelta(dt, 'h')
        for mode in modes:

            ts1 = ts_ins[mode].rename('insitu')
            ts2 = ts_smp[mode].rename('smap')
            ts3 = ts_asc[mode].rename('ascat')

            df = pd.concat((ts1, ts2), axis='columns').dropna()
            rs_smp.loc[dt,mode] = df.corr()['insitu']['smap']

            df = pd.concat((ts1, ts3), axis='columns').dropna()
            rs_asc.loc[dt,mode] = df.corr()['insitu']['ascat']

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    rs_smp.plot(ax=plt.gca())
    plt.axvline(color='k', linestyle='--', linewidth=1)
    plt.title('R(SMAP, in situ)')
    plt.xlabel('t(SMAP) - t(in situ) [hours]')

    plt.subplot(1, 2, 2)
    rs_asc.plot(ax=plt.gca())
    plt.axvline(color='k', linestyle='--', linewidth=1)
    plt.title('R(ASCAT, in situ)')
    plt.xlabel('t(ASCAT) - t(in situ) [hours]')


    plt.tight_layout()
    plt.show()


def satellite_experiment_tca():

    # network = 'RSMN'
    # station = 'Bacles'

    # network = 'SMOSMANIA'
    # station = 'Sabres'

    network = 'AMMA-CATCH'
    station = 'Belefoungou-Mid'

    io_ins = ISMN_io()
    io_smp = SMAP_io()
    io_asc = HSAF_io()

    lat = io_ins.list[(io_ins.list['network']==network) & (io_ins.list['station']==station)]['latitude'].values[0]
    lon = io_ins.list[(io_ins.list['network'] == network) & (io_ins.list['station'] == station)]['longitude'].values[0]

    plot_rs = True
    plot_ts = False
    # plot_rs = False
    # plot_ts = True

    ts_ins = io_ins.read(network, station, surface_only=True, surface_depth=0.05).dropna().resample('1h').mean()
    ts_ins.index.name = None
    ts_ins.columns = ['raw']
    ts_ins['clim'] = calc_anom(ts_ins['raw'], mode='climatological', return_clim=True)
    ts_ins['anom_clim'] = calc_anom(ts_ins['raw'], mode='climatological')
    ts_ins['anom_lt'] = calc_anom(ts_ins['raw'], mode='longterm')
    ts_ins['anom_st'] = calc_anom(ts_ins['raw'], mode='shortterm')

    ts_smp = io_smp.read(lat,lon)['soil_moisture']
    ts_smp = ts_smp.resample('1h').mean().dropna()
    ts_smp.name = 'raw'
    ts_smp = pd.DataFrame(ts_smp)
    ts_smp['clim'] = calc_anom(ts_smp['raw'], mode='climatological', return_clim=True)
    ts_smp['anom_clim'] = calc_anom(ts_smp['raw'], mode='climatological')
    ts_smp['anom_lt'] = calc_anom(ts_smp['raw'], mode='longterm')
    ts_smp['anom_st'] = calc_anom(ts_smp['raw'], mode='shortterm')

    ts_asc = io_asc.read(lat, lon, ascat_id='C')
    ts_asc = ts_asc.resample('1h').mean().dropna()
    ts_asc.name = 'raw'
    ts_asc = pd.DataFrame(ts_asc)
    ts_asc['clim'] = calc_anom(ts_asc['raw'], mode='climatological', return_clim=True)
    ts_asc['anom_clim'] = calc_anom(ts_asc['raw'], mode='climatological')
    ts_asc['anom_lt'] = calc_anom(ts_asc['raw'], mode='longterm')
    ts_asc['anom_st'] = calc_anom(ts_asc['raw'], mode='shortterm')

    modes = ['raw', 'anom_clim', 'anom_lt', 'anom_st']

    dts = np.arange(-24,25)

    rs_smp = pd.DataFrame(columns=modes, index=dts)
    rs_asc = pd.DataFrame(columns=modes, index=dts)

    idx = ts_ins.index
    for dt in dts:
        ts_ins.index = idx + pd.to_timedelta(dt, 'h')
        for mode in modes:

            ts1 = ts_ins[mode].rename('insitu')
            ts2 = ts_smp[mode].rename('smap')
            ts3 = ts_asc[mode].rename('ascat')

            df = pd.concat((ts1, ts2), axis='columns').dropna()
            rs_smp.loc[dt,mode] = df.corr()['insitu']['smap']

            df = pd.concat((ts1, ts3), axis='columns').dropna()
            rs_asc.loc[dt,mode] = df.corr()['insitu']['ascat']

    # plt.figure(figsize=(18, 8))
    #
    # plt.subplot(1, 3, 1)
    # rs_smp.plot(ax=plt.gca())
    # plt.axvline(color='k', linestyle='--', linewidth=1)
    # plt.title('R(SMAP, in situ)')
    # plt.xlabel('t(SMAP) - t(in situ) [hours]')
    #
    # plt.subplot(1, 3, 2)
    # rs_asc.plot(ax=plt.gca())
    # plt.axvline(color='k', linestyle='--', linewidth=1)
    # plt.title('R(ASCAT, in situ)')
    # plt.xlabel('t(ASCAT) - t(in situ) [hours]')
    #
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(14, 12))

    ts_ins.index = idx
    dts2 =  np.arange(24)
    for k, mode in enumerate(modes):

        ts1 = ts_ins[mode].rename('insitu')
        ts2 = ts_smp[mode].rename('smap')
        ts3 = ts_asc[mode].rename('ascat')
        tcrs = pd.DataFrame(columns=['insitu','smap','ascat'], index=dts2)

        for d in dts2:

            # Create reference time steps for the respective reference hour of the day of this iteration
            ref_df = pd.DataFrame(
                index=pd.date_range('2015-01-01', '2022-01-01') + pd.Timedelta(d, 'h'))

            # Find the NN to the reference time steps for each data set
            args = [ts1.dropna(), ts2.dropna(), ts3.dropna()]
            matched = df_match(ref_df, *args, window=0.5)
            for i, col in enumerate(['insitu','smap','ascat']):
                ref_df[col] = matched[i][col]
            ref_df.dropna(inplace=True)

            tcr, _, _ = TCA_calc(ref_df)
            tcrs.loc[d, 'insitu'] = tcr[0]
            tcrs.loc[d, 'smap'] = tcr[1]
            tcrs.loc[d, 'ascat'] = tcr[2]

        plt.subplot(2, 2, k+1)
        tcrs.plot(ax=plt.gca())
        plt.title(f'R ({mode})')
        if k > 1:
            plt.xlabel('ref time [hours]')
        plt.ylim(0.2, 1)
        for l in np.unique(ts_smp.index.hour):
            plt.axvline(l, color='orange', linestyle='--', linewidth=1)
        for l in np.unique(ts_asc.index.hour):
            plt.axvline(l, color='green', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


def insitu_experiment():

    # network = 'RSMN'
    # station = 'Bacles'

    # network = 'SMOSMANIA'
    # station = 'Sabres'

    network = 'AMMA-CATCH'
    station = 'Belefoungou-Mid'

    io = ISMN_io()
    surface_depth = 0.05

    # plot_rs = True
    # plot_ts = False
    plot_rs = False
    plot_ts = True

    ts = io.read(network, station, surface_only=True, surface_depth=surface_depth).dropna().resample('1h').mean()
    ts.index.name = None
    ts.columns = ['raw']

    ts['clim'] = calc_anom(ts['raw'], mode='climatological', return_clim=True)
    ts['anom_clim'] = calc_anom(ts['raw'], mode='climatological')
    ts['anom_lt'] = calc_anom(ts['raw'], mode='longterm')
    ts['anom_st'] = calc_anom(ts['raw'], mode='shortterm')

    sm = ts['anom_st'].values

    if plot_rs:

        # n_days = 50   # anom_lt
        n_days = 6   # anom_st

        nmin = -24 * n_days
        nmax = 24 * n_days
        # stepsize = 4
        steps = 49

        # dts2 = np.arange(nmin, nmax, stepsize)
        # dts3 = np.arange(nmin, nmax, stepsize)
        dts2 = np.linspace(nmin, nmax, steps).astype('int')
        dts3 = np.linspace(nmin, nmax, steps).astype('int')
        dtsx, dtsy = np.meshgrid(dts2,dts3)

        reps = 1

        shp = [len(dts2), len(dts3), reps]
        r12 = np.full(shp, np.nan)
        r13 = np.full(shp, np.nan)
        r23 = np.full(shp, np.nan)

        rmsd12 = np.full(shp, np.nan)
        rmsd13 = np.full(shp, np.nan)
        rmsd23 = np.full(shp, np.nan)

        r1 = np.full(shp, np.nan)
        r2 = np.full(shp, np.nan)
        r3 = np.full(shp, np.nan)

        err1 = np.full(shp, np.nan)
        err2 = np.full(shp, np.nan)
        err3 = np.full(shp, np.nan)

        r1_t = np.full(reps, np.nan)
        r2_t = np.full(reps, np.nan)
        r3_t = np.full(reps, np.nan)

        for rep in np.arange(reps):

            cnt = 0
            for ix, dt2 in enumerate(dts2-nmin):
                for iy, dt3 in enumerate(dts3-nmin):
                    cnt += 1
                    print(f'{cnt} / {len(dts2)*len(dts3)} ({rep+1})')

                    x1, x2, x3 = perturb_data(sm)

                    # t = dt.now()
                    # print((dt.now() - t).microseconds / 1e6)

                    ref = x1[-nmin:len(x1) - nmax]
                    nref = len(ref)

                    df = pd.DataFrame({0: sm, 1: x1, 2: x2, 3: x3})

                    corr = df.corr()**2
                    r1_t[rep] = corr[0][1]
                    r2_t[rep] = corr[0][2]
                    r3_t[rep] = corr[0][3]

                    tmp2 = x2[dt2:dt2+nref]
                    tmp3 = x3[dt3:dt3+nref]
                    df = pd.DataFrame({0:ref, 1:tmp2, 2:tmp3})

                    corr = df.corr()**2
                    r12[iy, ix, rep] = corr[0][1]
                    r13[iy, ix, rep] = corr[0][2]
                    r23[iy, ix, rep] = corr[1][2]

                    rmsd = calc_rmsd(df)
                    rmsd12[iy, ix, rep] = rmsd[0]
                    rmsd13[iy, ix, rep] = rmsd[1]
                    rmsd23[iy, ix, rep] = rmsd[2]

                    tcr, err, beta = TCA_calc(df)
                    err /= beta

                    r1[iy, ix, rep] = tcr[0]
                    r2[iy, ix, rep] = tcr[1]
                    r3[iy, ix, rep] = tcr[2]

                    err1[iy, ix, rep] = err[0]
                    err2[iy, ix, rep] = err[1]
                    err3[iy, ix, rep] = err[2]


        r1_t = r1_t.mean()
        r2_t = r2_t.mean()
        r3_t = r3_t.mean()

        r12 = r12.mean(axis=2)
        r13 = r13.mean(axis=2)
        r23 = r23.mean(axis=2)
        rmsd12 = rmsd12.mean(axis=2)
        rmsd13 = rmsd13.mean(axis=2)
        rmsd23 = rmsd23.mean(axis=2)

        r1 = r1.mean(axis=2)
        r2 = r2.mean(axis=2)
        r3 = r3.mean(axis=2)
        err1 = err1.mean(axis=2)
        err2 = err2.mean(axis=2)
        err3 = err3.mean(axis=2)

        f, ax = plt.subplots(3, 3, figsize=(14,12), sharey='all', sharex='all')

        vmin = 0.0
        vmax = 0.5

        ax[0, 0].pcolormesh(dtsx, dtsy, r12, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('R(x1, x2)')
        # ax[0, 0].set_xlabel('dt (x2)')
        ax[0, 0].set_ylabel('dt (x3) [hours]')

        ax[0, 1].pcolormesh(dtsx, dtsy, r13, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('R(x1, x3)')
        # ax[0, 1].set_xlabel('dt (x2)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r23, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 2].set_title('R(x2, x3)')
        # ax[0, 2].set_xlabel('dt (x2)')

        cb = f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0.2
        vmax = 1

        ax[1, 0].pcolormesh(dtsx, dtsy, r1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'R(x1)')
        ax[1, 0].set_ylabel('dt (x3) [hours]')

        ax[1, 1].pcolormesh(dtsx, dtsy, r2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'R(x2)')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, r3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'R(x3)')
        cb = f.colorbar(im, ax=ax[1, 2], orientation='vertical')

        vmin = 0
        vmax = 0.06

        ax[2, 0].pcolormesh(dtsx, dtsy, err1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[2, 0].set_title(f'RMSE(x1)')
        ax[2, 0].set_xlabel('dt (x2) [hours]')
        ax[2, 0].set_ylabel('dt (x3) [hours]')

        ax[2, 1].pcolormesh(dtsx, dtsy, err2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[2, 1].set_title(f'RMSE(x2)')
        ax[2, 1].set_xlabel('dt (x2) [hours]')

        im = ax[2, 2].pcolormesh(dtsx, dtsy, err3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[2, 2].set_title(f'RMSE(x3)')
        ax[2, 2].set_xlabel('dt (x2) [hours]')

        cb = f.colorbar(im, ax=ax[2, 2], orientation='vertical')

    if plot_ts:

        tau = estimate_tau(ts, n_lags=24*1)
        ts.columns = [f'{a} (tau={b/24:.1f} days)' for a,b in zip(ts.columns.values, tau)]
        plt.figure(figsize=(20,7))
        ts.plot(ax=plt.gca())
        plt.axhline(color='k', linestyle='--', linewidth=1)
        plt.title(f'{network} - {station}')

    plt.tight_layout()
    plt.show()

def plot_sig_est_decline():

    network = 'RSMN'
    station = 'Bacles'

    io = ISMN_io()
    surface_depth = 0.05

    ts = io.read(network, station, surface_only=True, surface_depth=surface_depth).dropna().resample('1h').mean()
    ts.index.name = None
    ts.columns = ['raw']

    ts['clim'] = calc_anom(ts['raw'], mode='climatological', return_clim=True)
    ts['anom_clim'] = calc_anom(ts['raw'], mode='climatological')
    ts['anom_lt'] = calc_anom(ts['raw'], mode='longterm')
    ts['anom_st'] = calc_anom(ts['raw'], mode='shortterm')

    sm = ts['anom_st'].values

    n_days = 5   #anom_st

    nmin = -24 * n_days
    nmax = -nmin
    stepsize = 1

    dts2 = np.arange(nmin, nmax, stepsize)
    dts3 = -np.arange(nmin, nmax, stepsize)

    cov12 = np.full(len(dts2), np.nan)
    cov13 = np.full(len(dts2), np.nan)
    cov23 = np.full(len(dts2), np.nan)
    sig = np.full(len(dts2), np.nan)

    for i, (dt2, dt3) in enumerate(zip(dts2-nmin, dts3-nmin)):

        x1, x2, x3 = perturb_data(sm)

        ref = x1[-nmin:len(x1) - nmax]
        nref = len(ref)

        tmp2 = x2[dt2:dt2+nref]
        tmp3 = x3[dt3:dt3+nref]
        df = pd.DataFrame({0:ref, 1:tmp2, 2:tmp3})

        cov = df.cov()

        cov12[i] = cov[0][1]
        cov13[i] = cov[0][2]
        cov23[i] = cov[1][2]
        sig[i] = cov[0][1] * cov[0][2] / cov[1][2]

    plt.figure(figsize=(20,10))


    plt.subplot(2, 1, 1)
    pd.DataFrame({'12,13':cov12, '23':cov23}, index=dts2).plot(ax=plt.gca())
    plt.xlim(0, nmax)
    # plt.ylim(-0.02, 0.06)
    plt.axhline(linestyle='-', color='k', linewidth=1)

    plt.subplot(2,1,2)
    plt.plot(dts2, sig)
    plt.xlim(0, nmax)
    plt.ylim(-0.02, 0.06)
    plt.axhline(linestyle='-', color='k',linewidth=1)

    plt.tight_layout()
    plt.show()

def global_analysis():

    nprocs = 8

    if nprocs == 1:
        run(nprocs, nprocs)
    else:
        p = Pool(nprocs)
        part = np.arange(nprocs) + 1
        parts = repeat(nprocs, nprocs)
        p.starmap(run, zip(part, parts))
        # p.map(run, parts)

    # merge_files(r'H:\work\experiments\temporal_matching_uncertainty\results\ASC_AMS_SMP_AD_12h', precision='%0.8f', delete=True)

def run(part, parts):

    io = interface()

    collocation_window = 1    # [days]

    dir_out = Path(r'H:\work\experiments\temporal_matching_uncertainty\results\ASC_AMS_SMP_AD_24h')
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    result_file =  dir_out / f'result_part{part}.csv'

    tmp_result_file =  dir_out / f'result.csv'
    if tmp_result_file.exists():
        tmp_res = pd.read_csv(tmp_result_file, index_col=0)
        diff = set(io.lut.index) - set(tmp_res.index)
        io.lut = io.lut.loc[list(diff)]

    subs = (np.arange(parts + 1) * len(io.lut) / parts).astype('int')
    subs[-1] = len(io.lut)
    start = subs[part - 1]
    end = subs[part]

    io.lut = io.lut.iloc[start:end,:]

    modes = ['raw', 'anom_clim', 'anom_lt', 'anom_st']
    dts = np.arange(24)

    res_cols = ['lat', 'lon', 'ease_col', 'ease_row'] + [f'n_{dt}'for dt in dts] + \
               [f'r_{mode}_{dt}_{io.names[0]}_{io.names[1]}' for dt in dts for mode in modes] + \
               [f'r_{mode}_{dt}_{io.names[0]}_{io.names[2]}' for dt in dts for mode in modes] + \
               [f'r_{mode}_{dt}_{io.names[1]}_{io.names[2]}' for dt in dts for mode in modes]

    for cnt, (dss, info) in enumerate(io.iter_gps()):

        print(f'{cnt} / {len(io.lut)}')

        if any([len(ds)<10 for ds in dss]):
            continue

        res = pd.DataFrame(columns=res_cols, index=(info.name,))
        res.loc[info.name, 'lat'] = info.lat
        res.loc[info.name, 'lon'] = info.lon
        res.loc[info.name, 'ease_col'] = info.ease_col
        res.loc[info.name, 'ease_row'] = info.ease_row

        for dt in dts:

            ref_df = pd.DataFrame(
                index=pd.date_range('2015-01-01', '2022-01-01') + pd.Timedelta(dt, 'h'))

            try:
                matched = combined_temporal_collocation(ref_df, dss, collocation_window, dropduplicates=True).dropna()
            except:
                continue

            if len(matched) < 10:
                continue

            res.loc[info.name, f'n_{dt}'] = len(matched)

            for mode in modes:
                tmp_df = matched[mode]
                tmp_df.columns = io.names

                corr = tmp_df.corr()

                res.loc[info.name, f'r_rel_{mode}_{dt}_{io.names[0]}_{io.names[1]}'] = corr[io.names[0]][io.names[1]]
                res.loc[info.name, f'r_rel_{mode}_{dt}_{io.names[0]}_{io.names[2]}'] = corr[io.names[0]][io.names[2]]
                res.loc[info.name, f'r_rel_{mode}_{dt}_{io.names[1]}_{io.names[2]}'] = corr[io.names[1]][io.names[2]]

                tcr, _, _ = TCA_calc(tmp_df)

                res.loc[info.name, f'r_tca_{mode}_{dt}_{io.names[0]}'] = tcr[0]**0.5
                res.loc[info.name, f'r_tca_{mode}_{dt}_{io.names[1]}'] = tcr[1]**0.5
                res.loc[info.name, f'r_tca_{mode}_{dt}_{io.names[2]}'] = tcr[2]**0.5

        if not result_file.exists():
            res.to_csv(result_file, float_format='%0.4f')
        else:
            res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

def generate_lut():

    fname = r"D:\data_sets\lut_smap_ascat_amsr2.csv"

    io = interface()

    lats = io.io_smp.lats
    lons = io.io_smp.lons

    cols = ['lat', 'lon', 'ease_row', 'ease_col', 'gpi_asc', 'cell_asc', 'gpi_ams', 'cell_ams']
    lut = pd.DataFrame(columns=cols, index=np.arange(len(lats)*len(lons)))

    i = -1
    for row, lat in enumerate(lats):
        for col, lon in enumerate(lons):
            i += 1
            print(f'{i} / {len(lut)}')

            gpi_asc = io.io_asc.grid.find_nearest_gpi(lon, lat, max_dist=18000)[0]
            if type(gpi_asc) != np.int32:
                continue
            if gpi_asc not in io.io_asc.grid.activegpis:
                continue
            cell_asc = io.io_asc.grid.activearrcell[io.io_asc.grid.activegpis==gpi_asc][0]

            gpi_ams = io.io_lprm.grid.find_nearest_gpi(lon, lat, max_dist=18000)[0]
            if type(gpi_ams) != np.int32:
                continue
            if gpi_ams not in io.io_lprm.grid.activegpis:
                continue
            cell_ams = io.io_lprm.grid.activearrcell[io.io_lprm.grid.activegpis == gpi_ams][0]

            lut.loc[i, 'lat'] = lat
            lut.loc[i, 'lon'] = lon
            lut.loc[i, 'ease_row'] = row
            lut.loc[i, 'ease_col'] = col
            lut.loc[i, 'gpi_asc'] = gpi_asc
            lut.loc[i, 'cell_asc'] = cell_asc
            lut.loc[i, 'gpi_ams'] = gpi_ams
            lut.loc[i, 'cell_ams'] = cell_ams

    lut = lut.dropna().sort_values('cell_ams', axis=0)
    lut.index = np.arange(len(lut))
    lut.to_csv(fname)

if __name__ == '__main__':

    # check_syn_ts()
    # satellite_experiment_rel()
    # satellite_experiment_tca()
    # insitu_experiment()
    # plot_sig_est_decline()
    global_analysis()
    # run(1)
    # generate_lut()