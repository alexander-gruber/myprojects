import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from myprojects.io.gen_syn_data import generate_soil_moisture, generate_error
from myprojects.io.insitu import ISMN_io
from myprojects.timeseries import calc_anom

import scipy.optimize as optimization

from validation_good_practice.ancillary.metrics import TCA_calc

def estimate_tau(in_df, n_lags=24*90):

    df = in_df.copy()
    n_cols = len(df.columns)

    # calculate auto-correlation coefficients for different lags [days]
    rho = np.full((n_cols, n_lags), np.nan)
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
            popt = None

    # plt.plot(rho[0,:])
    # plt.axhline(np.exp(1)**-1, color='k', linewidth=1, linestyle='--')
    # plt.axvline(tau[0], color='k', linewidth=1, linestyle='--')
    # plt.show()

    return tau, popt

def perturb_data(sm):

    size = len(sm)

    err1 = generate_error(size=size, var=np.nanvar(sm) / 2.)
    err2 = generate_error(size=size, var=np.nanvar(sm) / 2.)
    err3 = generate_error(size=size, var=np.nanvar(sm) / 2.)

    x1 = sm + err1
    x2 = sm + err2
    x3 = sm + err3

    return x1, x2, x3

def synthetic_experiment(dir_out):

    # network = 'RSMN'
    # station = 'Bacles'

    # network = 'SMOSMANIA'
    # station = 'Sabres'

    network = 'AMMA-CATCH'
    station = 'Belefoungou-Mid'

    io = ISMN_io()
    surface_depth = 0.05

    plot_relative = False
    plot_tca = True

    ts = io.read(network, station, surface_only=True, surface_depth=surface_depth).dropna().resample('1h').mean()
    ts.index.name = None
    ts.columns = ['raw']

    ts['clim'] = calc_anom(ts['raw'], mode='climatological', return_clim=True)
    ts['anom_clim'] = calc_anom(ts['raw'], mode='climatological')
    ts['anom_lt'] = calc_anom(ts['raw'], mode='longterm')
    ts['anom_st'] = calc_anom(ts['raw'], mode='shortterm')

    sm = ts['anom_st'].values

    # n_days = 50   # anom_lt
    n_days = 1   # anom_st

    nmin = -24 * n_days
    nmax = 24 * n_days
    steps = 49

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

                df = pd.DataFrame({0: sm, 1: x1, 2: x2, 3: x3}).dropna()

                corr = df.corr()
                r1_t[rep] = corr[0][1]
                r2_t[rep] = corr[0][2]
                r3_t[rep] = corr[0][3]

                tmp2 = x2[dt2:dt2+nref]
                tmp3 = x3[dt3:dt3+nref]
                df = pd.DataFrame({0:ref, 1:tmp2, 2:tmp3}).dropna()

                corr = df.corr()
                r12[iy, ix, rep] = corr[0][1]
                r13[iy, ix, rep] = corr[0][2]
                r23[iy, ix, rep] = corr[1][2]

                rmsd = lambda a, b: np.sqrt(np.nanmean((a - b) ** 2))
                rmsd12[iy, ix, rep] = rmsd(df[0], df[1])
                rmsd13[iy, ix, rep] = rmsd(df[0], df[2])
                rmsd23[iy, ix, rep] = rmsd(df[1], df[2])

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

    if plot_relative:
        f, ax = plt.subplots(2, 3, figsize=(14,9), sharey='all', sharex='all')

        vmin = 0.3
        vmax = 0.7

        ax[0, 0].pcolormesh(dtsx, dtsy, r12, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('R(x1, x2)')
        ax[0, 0].set_ylabel('dt (x3) [hours]')

        ax[0, 1].pcolormesh(dtsx, dtsy, r13, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('R(x1, x3)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r23, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 2].set_title('R(x2, x3)')

        f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0.03
        vmax = 0.05

        ax[1, 0].pcolormesh(dtsx, dtsy, rmsd12, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'RMSD(x1, x2)')
        ax[1, 0].set_ylabel('dt (x3) [hours]')
        ax[1, 0].set_xlabel('dt (x2)')

        ax[1, 1].pcolormesh(dtsx, dtsy, rmsd13, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'RMSD(x1, x3)')
        ax[1, 1].set_xlabel('dt (x2)')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, rmsd23, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'RMSD(x2, x3)')
        ax[1, 2].set_xlabel('dt (x2)')
        f.colorbar(im, ax=ax[1, 2], orientation='vertical')

        f.savefig(dir_out / 'syn_exp_rel.png', dpi=300, bbox_inches='tight')
        plt.close()

    if plot_tca:
        f, ax = plt.subplots(2, 3, figsize=(14,9), sharey='all', sharex='all')

        vmin = 0.3
        vmax = 0.7

        ax[0, 0].pcolormesh(dtsx, dtsy, r1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('R(x1)')
        ax[0, 0].set_ylabel('dt (x3) [hours]')

        ax[0, 1].pcolormesh(dtsx, dtsy, r2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('R(x2)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 2].set_title('R(x3)')

        f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0.01
        vmax = 0.035

        ax[1, 0].pcolormesh(dtsx, dtsy, err1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'RMSE(x1)')
        ax[1, 0].set_xlabel('dt (x2) [hours]')
        ax[1, 0].set_ylabel('dt (x3) [hours]')

        ax[1, 1].pcolormesh(dtsx, dtsy, err2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'RMSE(x2)')
        ax[1, 1].set_xlabel('dt (x2) [hours]')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, err3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'RMSE(x3)')
        ax[1, 2].set_xlabel('dt (x2) [hours]')

        cb = f.colorbar(im, ax=ax[1, 2], orientation='vertical')

        f.savefig(dir_out / 'syn_exp_tca.png', dpi=300, bbox_inches='tight')
        plt.close()

def synthetic_experiment_corrected(dir_out):

    # network = 'RSMN'
    # station = 'Bacles'

    # network = 'SMOSMANIA'
    # station = 'Sabres'

    network = 'AMMA-CATCH'
    station = 'Belefoungou-Mid'

    io = ISMN_io()
    surface_depth = 0.05

    plot_relative = True
    plot_tca = False

    ts = io.read(network, station, surface_only=True, surface_depth=surface_depth).dropna().resample('1h').mean()
    ts.index.name = None
    ts.columns = ['raw']

    ts['clim'] = calc_anom(ts['raw'], mode='climatological', return_clim=True)
    ts['anom_clim'] = calc_anom(ts['raw'], mode='climatological')
    ts['anom_lt'] = calc_anom(ts['raw'], mode='longterm')
    ts['anom_st'] = calc_anom(ts['raw'], mode='shortterm')

    sm = ts['anom_st'].values

    n_lags = 10*24

    # x1, x2, x3 = perturb_data(sm)
    # tau, popt = estimate_tau(pd.DataFrame(sm, index=ts.index), n_lags=n_lags)
    # r_corr = lambda t: np.exp(popt[0]*t)


    # n_days = 50   # anom_lt
    n_days = 1   # anom_st

    nmin = -24 * n_days
    nmax = 24 * n_days
    steps = 49

    r_corr = lambda dt1, dt2: pd.DataFrame({0:sm[dt1:dt1 + nref], 1:sm[dt2:dt2 + nref]}).dropna().corr()[0][1]

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

                ref = x1[-nmin:len(x1) - nmax]
                nref = len(ref)

                df = pd.DataFrame({0: sm, 1: x1, 2: x2, 3: x3}).dropna()

                corr = df.corr()
                r1_t[rep] = corr[0][1]
                r2_t[rep] = corr[0][2]
                r3_t[rep] = corr[0][3]

                tmp2 = x2[dt2:dt2+nref]
                tmp3 = x3[dt3:dt3+nref]

                df = pd.DataFrame({0:ref, 1:tmp2, 2:tmp3}).dropna()

                corr = df.corr()
                r12[iy, ix, rep] = corr[0][1] / r_corr(-nmin, dt2)
                r13[iy, ix, rep] = corr[0][2] / r_corr(-nmin, dt3)
                r23[iy, ix, rep] = corr[1][2] / r_corr(dt2, dt3)

                msd = lambda a, b: np.nanmean((a - b) ** 2)
                cov = df.cov()
                rmsd12[iy, ix, rep] = np.sqrt( msd(df[0], df[1]) - 2*cov[0][1] * (1 / r_corr(-nmin, dt2) - 1))
                rmsd13[iy, ix, rep] = np.sqrt( msd(df[0], df[2]) - 2*cov[0][2] * (1 / r_corr(-nmin, dt3) - 1))
                rmsd23[iy, ix, rep] = np.sqrt( msd(df[1], df[2]) - 2*cov[1][2] * (1 / r_corr(dt2, dt3) - 1))

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

    if plot_relative:
        f, ax = plt.subplots(2, 3, figsize=(14,9), sharey='all', sharex='all')

        vmin = 0.3
        vmax = 0.7

        ax[0, 0].pcolormesh(dtsx, dtsy, r12, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('R(x1, x2)')
        ax[0, 0].set_ylabel('dt (x3) [hours]')

        ax[0, 1].pcolormesh(dtsx, dtsy, r13, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('R(x1, x3)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r23, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 2].set_title('R(x2, x3)')

        f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0.03
        vmax = 0.05

        ax[1, 0].pcolormesh(dtsx, dtsy, rmsd12, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'RMSD(x1, x2)')
        ax[1, 0].set_ylabel('dt (x3) [hours]')
        ax[1, 0].set_xlabel('dt (x2)')

        ax[1, 1].pcolormesh(dtsx, dtsy, rmsd13, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'RMSD(x1, x3)')
        ax[1, 1].set_xlabel('dt (x2)')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, rmsd23, cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'RMSD(x2, x3)')
        ax[1, 2].set_xlabel('dt (x2)')
        f.colorbar(im, ax=ax[1, 2], orientation='vertical')

        f.savefig(dir_out / 'syn_exp_rel_corr.png', dpi=300, bbox_inches='tight')
        plt.close()

        # plt.tight_layout()
        # plt.show()

    if plot_tca:
        f, ax = plt.subplots(2, 3, figsize=(14,9), sharey='all', sharex='all')

        vmin = 0.2
        vmax = 1

        ax[0, 0].pcolormesh(dtsx, dtsy, r1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 0].set_title('R(x1)')
        ax[0, 0].set_ylabel('dt (x3) [hours]')

        ax[0, 1].pcolormesh(dtsx, dtsy, r2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('R(x2)')

        im = ax[0, 2].pcolormesh(dtsx, dtsy, r3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 2].set_title('R(x3)')

        f.colorbar(im, ax=ax[0, 2], orientation='vertical')

        vmin = 0
        vmax = 0.06

        ax[1, 0].pcolormesh(dtsx, dtsy, err1, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 0].set_title(f'RMSE(x1)')
        ax[1, 0].set_xlabel('dt (x2) [hours]')
        ax[1, 0].set_ylabel('dt (x3) [hours]')

        ax[1, 1].pcolormesh(dtsx, dtsy, err2, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title(f'RMSE(x2)')
        ax[1, 1].set_xlabel('dt (x2) [hours]')

        im = ax[1, 2].pcolormesh(dtsx, dtsy, err3, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 2].set_title(f'RMSE(x3)')
        ax[1, 2].set_xlabel('dt (x2) [hours]')

        cb = f.colorbar(im, ax=ax[1, 2], orientation='vertical')

        f.savefig(dir_out / 'syn_exp_tca.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__=='__main__':

    dir_out = Path(r'H:\work\experiments\temporal_matching_uncertainty\plots\publication')

    synthetic_experiment(dir_out)
    # synthetic_experiment_corrected(dir_out)