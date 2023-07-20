import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

import seaborn as sns
sns.set_context('talk', font_scale=0.6)

import matplotlib.pyplot as plt
import colorcet as cc

from pytesmo.metrics import tcol_metrics as tcol
from pytesmo.df_metrics import tcol_metrics as df_tcol

from myprojects.experiments.seasonal_tca.tca_adapter import doy_apply, calc_err_snr

def get_subset(df, doy, window_size=90):

    doys = df.index.dayofyear
    lower = round(doy - window_size/2)
    upper = round(doy + window_size/2)
    if lower >= 1:
        if upper <= 366:
            return df[(doys >= lower) & (doys <= upper)]
        else:
            return df[(doys >= lower) | (doys <= upper - 366)]
    else:
        return df[(doys <= upper) | (doys >= 366 - abs(lower))]

def static_tca(df):

    tc = df_tcol(df)
    err = tc[1][0]._asdict()
    beta = tc[2][0]._asdict()
    return [err[k] / beta[k] for k in err]

def seasonal_tca(df):

    res = np.full((366, 3), np.nan)
    for doy in np.arange(1,366):
        tmp_df = get_subset(df, doy)
        res[doy-1, :] = static_tca(tmp_df)
    return res

def test_static():

    n_yr = 10 # years
    n_ds = 350 # days / yr

    n = n_yr*n_ds

    sig = np.random.normal(0,10, size=n)
    t = np.arange(len(sig))

    per1 = 3 * np.cos(2 * np.pi * t / n_ds) + 5
    per2 = 3 * np.cos(2 * np.pi * t / n_ds) + 10
    per3 = 3 * np.cos(2 * np.pi * t / n_ds) + 15

    n_iters = 300

    res = pd.DataFrame(columns=['e1', 'e2', 'e3'], index=np.arange(n_iters), dtype='float')

    for i in np.arange(n_iters):

        ts1 = sig + np.random.normal(np.zeros(n), per1, size=n)
        ts2 = sig + np.random.normal(np.zeros(n), per2, size=n)
        ts3 = sig + np.random.normal(np.zeros(n), np.ones(n)*15, size=n)

        snr, err, beta = tcol(ts1, ts2, ts3)
        err /= beta
        res.loc[i, :] = err

    res.boxplot(figsize=(10,8))

    plt.axhline(per1.mean(), color='red', linestyle='--', linewidth=1)
    plt.axhline(per2.mean(), color='red', linestyle='--', linewidth=1)
    plt.axhline(per3.mean(), color='blue', linestyle='--', linewidth=1)
    plt.grid(False)

    plt.tight_layout()
    plt.show()

def test_dynamic_lineplot():

    dt = pd.date_range('2000-01-01', '2004-12-31', freq='D')

    n = len(dt)

    sig = np.random.normal(0, 10, size=n)

    per1 = 3 * np.sin(2 * np.pi * np.arange(366) / 366) + 8
    per2 = 3 * np.cos(2 * np.pi * np.arange(366) / 366) + 10
    errstd3 = 12

    n_iters = 300

    res = pd.DataFrame(columns=['doy', 'err_std', 'error'], dtype='float')
    tmp_res = pd.DataFrame(columns=['doy', 'err_std', 'error'], index = np.arange(1, 367), dtype='float')
    tmp_res.loc[:, 'doy'] = np.arange(1,367)

    for i in np.arange(n_iters):
        print(f'%i / %i' % (i+1, n_iters))

        Ser1 = pd.Series(sig.copy(), index=dt)
        Ser2 = pd.Series(sig.copy(), index=dt)
        Ser3 = pd.Series(sig.copy() + np.random.normal(np.zeros(n), np.ones(n) * errstd3, size=n), index=dt)
        for yr in np.unique(dt.year):
            if Ser1[Ser1.index.year==yr].index[0].is_leap_year:
                Ser1[Ser1.index.year==yr] +=  np.random.normal(np.zeros(366), per1)
                Ser2[Ser2.index.year==yr] +=  np.random.normal(np.zeros(366), per2)
            else:
                Ser1[Ser1.index.year == yr] += np.random.normal(np.zeros(365), per1[0:-1])
                Ser2[Ser2.index.year == yr] += np.random.normal(np.zeros(365), per2[0:-1])

        df = pd.concat((Ser1,Ser2,Ser3), axis='columns')
        df.columns = ['x', 'y', 'z']

        err_dyn = doy_apply(calc_err_snr, df)

        for j, ds in enumerate(['x','y','z']):
            tmp_res.loc[:, 'err_std'] = ds
            tmp_res.loc[:, 'error'] = err_dyn[:,j]

            res = res.append(tmp_res)


    res.index = np.arange(len(res))
    sns.lineplot(data=res, x="doy", y="error", hue="err_std")

    df = pd.DataFrame({'x_true': per1, 'y_true': per2, 'z_true':  np.ones(366) * errstd3})
    df.plot(ax=plt.gca(), linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


def test_dynamic_boxplot():

    dt = pd.date_range('2000-01-01', '2009-12-31', freq='D')

    n = len(dt)

    sig = np.random.normal(0, 10, size=n)

    per1 = 3 * np.sin(2 * np.pi * np.arange(366) / 366) + 8
    per2 = 3 * np.cos(2 * np.pi * np.arange(366) / 366) + 10
    errstd3 = 12

    n_iters = 300

    res = pd.DataFrame(columns=['x_true', 'y_true', 'z_true', 'x_static', 'y_static', 'z_static', 'x_dynamic', 'y_dynamic', 'z_dynamic'] , index = np.arange(n_iters), dtype='float')

    for i in np.arange(n_iters):
        print(f'%i / %i' % (i+1, n_iters))

        Ser1 = pd.Series(sig.copy(), index=dt)
        Ser2 = pd.Series(sig.copy(), index=dt)
        Ser3 = pd.Series(sig.copy() + np.random.normal(np.zeros(n), np.ones(n) * errstd3, size=n), index=dt)
        for yr in np.unique(dt.year):
            if Ser1[Ser1.index.year==yr].index[0].is_leap_year:
                Ser1[Ser1.index.year==yr] +=  np.random.normal(np.zeros(366), per1)
                Ser2[Ser2.index.year==yr] +=  np.random.normal(np.zeros(366), per2)
            else:
                Ser1[Ser1.index.year == yr] += np.random.normal(np.zeros(365), per1[0:-1])
                Ser2[Ser2.index.year == yr] += np.random.normal(np.zeros(365), per2[0:-1])

        df = pd.concat((Ser1,Ser2,Ser3), axis='columns')
        df.columns = ['x', 'y', 'z']

        err_true = [(df[col].values - sig).var() for col in df]

        err_stat = static_tca(df)
        err_dyn = doy_apply(calc_err_snr, df)

        res.loc[i, ['x_true', 'y_true', 'z_true']] = err_true
        res.loc[i, ['x_static', 'y_static', 'z_static']] = [e**2 for e in err_stat]
        res.loc[i, ['x_dynamic', 'y_dynamic', 'z_dynamic']] = np.nanmean(err_dyn**2, axis=0)

    res.boxplot(figsize=(10, 8))

    plt.axhline((per1**2).mean(), color='red', linestyle='--', linewidth=1)
    plt.axhline((per2**2).mean(), color='red', linestyle='--', linewidth=1)
    plt.axhline(errstd3**2, color='blue', linestyle='--', linewidth=1)
    plt.grid(False)

    plt.tight_layout()
    plt.show()


def test_simple():

    n_yr = 300  # years
    n_ds = 300  # days / yr

    n = n_yr * n_ds
    t = np.arange(n)

    per = 3 * np.cos(2 * np.pi * t / n_ds) + 5
    err = np.random.normal(np.zeros(n), per, size=n)

    per **= 2

    res = pd.DataFrame({'true_dyn': per[0:300],
                        'true_avg': np.ones(n_yr) * per.mean(),
                        'est_dyn1': err.reshape((300,300)).var(axis=0),
                        'est_dyn2': err.reshape((300,300)).var(axis=1),
                        'est_stat': err.var(),
                        })


    res.plot()
    plt.tight_layout()
    plt.show()

    # plt.plot(per)

    # plt.show()


if __name__=='__main__':

    test_dynamic_boxplot()