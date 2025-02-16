
import numpy as np
import pandas as pd

import logging

from pytesmo.time_series.anomaly import calc_climatology
from pytesmo.time_series.anomaly import calc_anomaly as calc_anom_pytesmo

def calc_anom(Ser, mode='climatological', window_size=35, return_clim=False, return_clim366=False):
    '''
    :param Ser:             pandas.Series; index must be a datetime index
    :param mode:            string; one of:
                                "climatological": calculate anomalies from the mean seasonal cycle
                                "longterm": inter-annual variabilities only (climatological anomalies minus short-term anomalies)
                                "shortterm": residuals from the seasonality (i.e., moving average) of each individual year
    :param window_size:     integer; window size for calculating the climatology and/or seasonality
    :param return_clim:     boolean; If true, the climatology value is returned for each timestep of the input Series
                                     This overrules the "mode" keyword!
    :param return_clim366:  boolean; If true, the actual climatology is returned (366 values)
                                     This overrules both the "mode" and "return_clim" keywords!
    '''

    if mode not in ['climatological', 'longterm', 'shortterm']:
        logging.error('calc_anom: unknown anomaly type')
        return None

    # Calculate the climatology
    if (mode != 'shortterm') | return_clim | return_clim366:
        clim = calc_climatology(Ser, respect_leap_years=True, wraparound=True, moving_avg_clim=window_size, fillna=False)
    else:
        clim = None

    # Return the actual climatology (366 values)
    if return_clim366:
        return clim

    # Calculate either climatological or short-term anomalies
    res = calc_anom_pytesmo(Ser, climatology=clim, window_size=window_size, return_clim=return_clim)

    # Derive long-term anomalies by subtracting short-term anomalies from climatological anomalies
    if (mode == 'longterm') and not return_clim:
        res -= calc_anom_pytesmo(Ser, climatology=None, window_size=window_size)

    # Return climatology values for each time step of the input Series
    if return_clim:
        res = res['climatology']

    try:
        res.name = Ser.name
    except:
        pass

    return res

def calc_anomaly(Ser, method='moving_average', output='anomaly', longterm=False, window_size=35, n=3):

    if (output=='climatology')&(longterm is True):
        output = 'climSer'

    xSer = Ser.dropna().copy()
    if len(xSer) == 0:
        return xSer

    doys = xSer.index.dayofyear.values
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1
    climSer = pd.Series(index=xSer.index)

    if not method in ['harmonic','mean','moving_average','ma']:
        logging.error('Unknown method: ' + method)
        return climSer

    if longterm is True:
        if method=='harmonic':
            clim = calc_clim_harmonic(xSer, n=n)
        if method=='mean':
            clim = calc_clim_harmonic(xSer, n=0)
        if (method=='moving_average')|(method=='ma'):
            clim = calc_clim_moving_average(xSer, window_size=window_size)
        if output == 'climatology':
            return clim
        climSer[:] = clim[doys]

    else:
        years = xSer.index.year
        for yr in np.unique(years):
            if method == 'harmonic':
                clim = calc_clim_harmonic(xSer[years == yr], n=n)
            if method == 'mean':
                clim = calc_clim_harmonic(xSer[years == yr], n=0)
            if (method == 'moving_average') | (method == 'ma'):
                clim = calc_clim_moving_average(xSer[years == yr], window_size=window_size)
            climSer[years == yr] = clim[doys[years == yr]].values

    if output == 'climSer':
        return climSer.reindex(Ser.index)

    climSer.name = xSer.name
    return xSer - climSer


def calc_clim_harmonic(Ser, n=3, cutoff=False):
    """
    Calculates the mean seasonal cycle of a data set
    by fitting harmonics.
    (!! Leap years are not yet properly treated !!)

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    n : int (optional)
        Number of harmonics that should be fitted.
        n=0 : long term mean
        n=1 : long term mean + annual cycle
        n=2 : long term mean + annual + half-annual cycle
        n=3 : long term mean + annual + half-annual + seasonal cycle
    cutoff : boolean
        If set, the climatology is not allowed to exceed the min/max of the original time series.

    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    """

    T = 365

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys>59)] -= 1

    A = np.ones((len(doys),2*n+1))

    for j in np.arange(n)+1:
        A[:,j] = np.cos(j * 2 * np.pi * doys / T)
        A[:,j+n] = np.sin(j * 2 * np.pi * doys / T)

    A = np.matrix(A)
    y = np.matrix(xSer.values).T
    try:
        x = np.array((A.T * A).I * A.T * y).flatten()
    except:
        x = np.full(2*n+1,np.nan)

    doys = np.arange(T)+1
    clim = pd.Series(index=np.arange(T)+1)
    clim[:] = x[0]
    for j in np.arange(n)+1:
        clim[:] += x[j] * np.cos(j * 2 * np.pi * doys / T) + x[j+n] * np.sin(j * 2 * np.pi * doys / T)

    if (cutoff is True)&(len(clim.dropna()!=0)):
        p = np.nanpercentile(xSer.values, [5,95])
        clim[(clim<p[0])|(clim>p[1])] = np.nan

    return clim

def calc_clim_moving_average(Ser, window_size=35, n_min=5, return_n=False):
    """
    Calculates the mean seasonal cycle as long-term mean within a moving average window.

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    window_size : int
        Moving Average window size
    n_min : int
        Minimum number of data points to calculate average
    return_n : boolean
        If true, the number of data points over which is averaged is returned
    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    clim_doys =  np.arange(365) + 1
    clim = pd.Series(index=clim_doys)
    n_data = pd.Series(index=clim_doys)

    for doy in clim_doys:

        # Avoid artifacts at start/end of year
        tmp_doys = doys.copy()
        if doy < window_size/2.:
            tmp_doys[tmp_doys > 365 - (np.ceil(window_size/2.)-doy)] -= 365
        if doy > 365 - (window_size/2. - 1):
            tmp_doys[tmp_doys < np.ceil(window_size/2.) - (365-doy)] += 365

        n_data[doy] = len(xSer[(tmp_doys >= doy - np.floor(window_size/2.)) & \
                               (tmp_doys <= doy + np.floor(window_size/2.))])

        if n_data[doy] >= n_min:
            clim[doy] = xSer[(tmp_doys >= doy - np.floor(window_size/2.)) & \
                             (tmp_doys <= doy + np.floor(window_size/2.))].values.mean()

    if return_n is False:
        return clim
    else:
        return clim, n_data


def calc_clim_p(ts, mode='pentadal', n=3):

    if mode == 'pentadal':
        clim = calc_pentadal_mean(ts)
    else:
        clim = calc_clim_harmonic(ts, n=n)
        pentads = np.floor((clim.index.values - 1) / 5.)
        clim = clim.groupby(pentads,axis=0).mean()
        clim.index = np.arange(73)+1

    return clim


def calc_pentadal_mean_std(Ser, n_min=9, return_n=False):
    """
    Calculates the mean seasonal cycle as long-term mean within a 45 days moving average window
    for each pentad (Faster than "calc_clim_moving_average" because output only per pentad)

    Parameters
    ----------
    Ser : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    n_min : int
        Minimum number of data points to calculate average
    return_n : boolean
        If true, the number of data points over which is averaged is returned
    Returns
    -------
    clim : pd.Series
        climatology of Ser (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xSer = Ser.dropna().copy()
    doys = xSer.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xSer.index.is_leap_year & (doys > 59)] -= 1

    Ser_pentad = np.floor((doys - 1) / 5.) + 1

    pentads = np.arange(73) + 1
    clim_mean = pd.Series(index=pentads)
    clim_std = pd.Series(index=pentads)
    n_data = pd.Series(index=pentads)
    for p in pentads:
        tmp_pentad = Ser_pentad.copy()
        if p < 5:
            tmp_pentad[tmp_pentad > 10] -= 73
        if p > 69:
            tmp_pentad[tmp_pentad < 60] += 73
        n_data[p] = len(xSer[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)])

        if n_data[p] >= n_min:
            clim_mean[p] = xSer[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)].values.mean()
            clim_std[p] = xSer[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)].values.std()

    # --- Time series are returned per pentad as needed for creating LDASSa scaling files!!
    # --- The following can map it to 365 values
    # doys = np.arange(1, 366).astype('int')
    # ind = np.floor((doys - 1) / 5.).astype('int') + 1
    # clim365 = pd.Series(clim_fcst.loc[ind].values, index=doys)

    if return_n is False:
        return clim_mean, clim_std
    else:
        return clim_mean, clim_std, n_data

