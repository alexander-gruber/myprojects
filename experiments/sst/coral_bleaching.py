import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

import seaborn as sns
sns.set_context('talk', font_scale=0.9)

from matplotlib import colors, colormaps
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

import cartopy.feature as cfeature
import cartopy.crs as ccrs

from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

from multiprocessing import Pool
from itertools import repeat

from netCDF4 import Dataset

from rasterio.features import geometry_mask
from affine import Affine
import geopandas as gpd

from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
from scipy.sparse import coo_array, dok_array, eye_array, diags_array, vstack
from scipy.sparse.linalg import spsolve, inv

from myprojects.timeseries import calc_anom
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula
import itertools




def create_mask():

    ds = xr.open_dataset(r"D:\data_sets\SST\v3.0.1\2000\01\01\20000101120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc")
    fout = r"D:\data_sets\SST\GBR\mask.nc"

    # Example variables, replace these with your actual data
    lon = ds.coords['lon'].values
    lat = ds.coords['lat'].values

    # Calculate resolution (assuming uniform grid)
    resolution_x = (lon[-1] - lon[0]) / (len(lon) - 1)
    resolution_y = (lat[-1] - lat[0]) / (len(lat) - 1)

    # Calculate the origin
    x_origin = lon[0] - (resolution_x / 2)
    y_origin = lat[0] - (resolution_y / 2)

    # Create the affine transform
    transform = Affine.translation(x_origin, y_origin) * Affine.scale(resolution_x, resolution_y)

    sf = gpd.read_file(r"D:\data_sets\SST\GBR\worldheritagemarineprogramme.shp")

    ShapeMask = geometry_mask(sf.iloc[0].drop('area_km2'),
                            out_shape=(len(ds.lat), len(ds.lon)),
                            transform=transform,
                            invert=True)

    ShapeMask = xr.DataArray(ShapeMask, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name='mask')

    ShapeMask.to_netcdf(fout)

def plot_avg_img():

    mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    files = sorted(Path(r"D:\data_sets\SST\v3.0.1\2000").glob("**/*.nc"))
    data = xr.open_mfdataset(files)

    latmax, lonmin = -9.767358414410054, 140.71507974440357
    latmin, lonmax = -25.896416782692228, 154.77144952418888
    extent = (lonmin, lonmax, latmin, latmax)

    mask_subs = mask.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
    subs = data.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax)).where(mask_subs)
    avg = subs['analysed_sst'].mean(dim='time').values - 273.15

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_title("Analysed Sea Surface Temperature (Celcius)")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    img = ax.imshow(np.flipud(avg), extent=extent, cmap='viridis', #vmin=lim[0], vmax=lim[1],
                    transform=ccrs.PlateCarree())

    # https://www.marineregions.org/gazetteer.php?p=details&id=26847
    fname = r"D:\data_sets\SST\GBR\worldheritagemarineprogramme.shp"
    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), facecolor='none', )
    ax.add_feature(shape_feature, linewidth=2, edgecolor='r')

    cbar = fig.colorbar(img, ax=fig.axes, orientation='vertical')

    # plt.tight_layout()
    plt.show()

def plot_avg_ts():

    # mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    # files = sorted(Path(r"D:\data_sets\SST\v3.0.1").glob("**/*.nc"))
    # data = xr.open_mfdataset(files)
    #
    # latmax, lonmin = -9.767358414410054, 140.71507974440357
    # latmin, lonmax = -25.896416782692228, 154.77144952418888
    #
    # mask_subs = mask.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
    # subs = data.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax)).where(mask_subs)
    # avg = subs['analysed_sst'].mean(dim=['lat','lon']) - 273.154

    data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")

    avg = data['sst'].mean(dim='loc')

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Analysed Sea Surface Temperature (Celcius)")

    avg.to_series().plot(ax=ax)
    plt.axhline(27, linewidth=1, linestyle='--', color='r')
    plt.axhline(28, linewidth=1, linestyle='--', color='r')
    plt.axhline(29, linewidth=1, linestyle='--', color='r')

    plt.tight_layout()
    plt.show()

def event_counter(data, threshold=27, min_dur=2):
    above_threshold = False
    events_count = 0
    duration = 0

    # Iterate through the time series
    for value in data:
        if value > threshold:
            if not above_threshold:
                # Start of a potential new event
                above_threshold = True
                duration = 1
            else:
                # Continuation of an event
                duration += 1
        else:
            if above_threshold and duration >= min_dur:
                # End of a valid event
                events_count += 1
            above_threshold = False
            duration = 0  # Reset the duration for the next potential event

    # Check if the last segment qualifies as an event
    if above_threshold and duration >= min_dur:
        events_count += 1

    return events_count

def plot_bleaching_events():

    data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")
    data = data['sst'].mean(dim='loc')
    # data['date'] += pd.Timedelta('180d')
    data['date'] = data['date'] - pd.Timedelta('183d')

    yrs = np.unique(data['date'].dt.year)
    yr_label = [f"{yr} / {yr+1}" for yr in yrs]
    event_lengths = [5, 20, 35]
    temperature_thresholds = [27, 28, 29]

    resarr = np.zeros((len(yrs), len(event_lengths), len(temperature_thresholds)))
    legend_titles = [f'{x} days' for x in event_lengths]

    for i, yr in enumerate(yrs):
        for j, dur in enumerate(event_lengths):
            for k, th in enumerate(temperature_thresholds):
                resarr[i, j, k] = event_counter(data[data['date'].dt.year == yr].values, threshold=th, min_dur=dur)

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(15, 10))
    for i, th in enumerate(temperature_thresholds):

        res = pd.DataFrame(resarr[:,:,i], index=yr_label, columns=legend_titles)
        res.name = 'Minimum event duration'

        axs[i] = plt.subplot(3, 1, i+1)
        axs[i].set_title(f"# Events where SST > {th} deg. C")
        if i < 2:
            res.plot.bar(stacked=True, ax=axs[i], legend=False)
        else:
            res.plot.bar(stacked=True, ax=axs[i])
            axs[i].legend(title='Min. duration')
            plt.xticks(rotation=70)

    plt.tight_layout()
    plt.show()

def extract_gbr_sst_unc():

    fout = r"D:\data_sets\SST\GBR\sst_gbr.nc"

    mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    files = sorted(Path(r"D:\data_sets\SST\v3.0.1").glob("**/*.nc"))

    idx = pd.DatetimeIndex([f.name[:14] for f in files])
    tmp_res = np.full((len(idx), len(np.where(mask.data)[0]), 2),np.nan)

    for i, f in enumerate(files):
        print(f'{i} / {len(files)}')

        try:
            with Dataset(f) as ds:
                tmp_res[i,:, 0] = ds['analysed_sst'][0,:,:][mask] - 273.15
                tmp_res[i,:, 1] = ds['analysed_sst_uncertainty'][0,:,:][mask]
        except:
            continue

    tmp_res[tmp_res<0] = np.nan

    res = xr.Dataset(data_vars = dict(sst=(["date", "loc"], tmp_res[:,:,0]),
                                      sst_uncertainty=(["date", "loc"], tmp_res[:,:,1])),
                     coords =  dict(date=idx,
                                    loc=np.arange(tmp_res.shape[1])))

    res.to_netcdf(fout)

# def extract_gbr_sst_unc2():
    #
    # fout = r"D:\data_sets\SST\GBR\sst_gbr2.nc"
    #
    # mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    # files = sorted(Path(r"D:\data_sets\SST\v3.0.1").glob("**/*.nc"))
    # data = xr.open_mfdataset(files)
    #
    # latmax, lonmin = -9.767358414410054, 140.71507974440357
    # latmin, lonmax = -25.896416782692228, 154.77144952418888
    #
    # mask_subs = mask.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))
    # subs = data.sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax))[['analysed_sst','analysed_sst_uncertainty']].where(mask_subs)
    #
    # encoding = {v: {'zlib': True, 'complevel': 5} for v in
    #             list(subs.data_vars.keys())}
    #
    # subs.to_netcdf(fout, encoding=encoding)
    # subs.close()


# def gauss(x, mu, sigma):
#     return np.exp(-(x-mu)**2/2/sigma**2)

def gauss(x, mean, std): # !!! Much faster to integrate over this than over scipy.stats.norm !!!
    return np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

class bimodal_gaussian(object):

    def __init__(self, mu1, sigma1, w1, mu2, sigma2, w2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.w1 = w1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.w2 = w2

    def pdf(self, x):
        return self.w1 * gauss(x, self.mu1, self.sigma1) + self.w2 * gauss(x, self.mu2, self.sigma2)

    def cdf(self, x):
        return np.array([quad(self.pdf, -np.inf, y)[0] for y in x.flatten()])

def fit_bimodal():

    # data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")
    # data = data['sst'].mean(dim='loc').to_series().values
    data = pd.read_csv(r"D:\data_sets\SST\GBR\gbr_sst_avg.csv", index_col=0).values

    # data['date'] = data['date'] - pd.Timedelta('183d')

    gmm = GaussianMixture(n_components=2)
    gmm.fit(data.reshape(-1, 1))

    means = gmm.means_

    # Conver covariance into Standard Deviation
    standard_deviations = gmm.covariances_ ** 0.5

    # Useful when plotting the distributions later
    weights = gmm.weights_

    print(f"Means: {means.flatten()}, Standard Deviations: {standard_deviations.flatten()}, Weights: {weights.flatten()}")


    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(6.4, 7))
    axes[0].hist(data, bins=50, alpha=0.5, density=True)
    x = np.linspace(min(data), max(data), 100)

    args = [item for t in zip(means.flatten(),standard_deviations.flatten(), weights) for item in t]
    bimodal = bimodal_gaussian(*args)
    bmpdf = bimodal.pdf(x)
    bmcdf = bimodal.cdf(x)

    for mean, std, weight in zip(means, standard_deviations, weights):
        pdf = weight * norm.pdf(x, mean, std)
        plt.plot(x.reshape(-1, 1), pdf.reshape(-1, 1), alpha=0.5)

    plt.plot(x.reshape(-1, 1), bmpdf.reshape(-1, 1), alpha=0.5)
    plt.plot(x.reshape(-1, 1), bmcdf.reshape(-1, 1), alpha=0.5)


    plt.tight_layout()
    plt.show()

def plot_prior_evidenec():

    data = pd.read_csv(r"D:\data_sets\SST\GBR\gbr_sst_avg.csv", index_col=0, parse_dates=True) * -1

    # detrend and remove seasonality
    x = np.arange(len(data))
    slope, intercept = np.polyfit(x, data, 1)
    detrended = pd.Series(data.values.flatten() - (slope * x),index=data.index)
    clim = calc_anom(detrended,return_clim366=True)
    # anom = detrended - clim

    data = clim

    # Estimate marginal PDF of the data
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data.values.reshape(-1, 1))
    means = gmm.means_
    standard_deviations = gmm.covariances_ ** 0.5
    weights = gmm.weights_
    args = [item for t in zip(means.flatten(),standard_deviations.flatten(), weights) for item in t]
    bimodal = bimodal_gaussian(*args)

    x = data.values.flatten()


    fig = plt.figure(figsize=(10,8))
    # n_lags = 2
    # arr = np.vstack([x[i:-(n_lags-i)] for i in np.arange(n_lags)]).T
    # theta = GumbelCopula(k_dim=n_lags).fit_corr_param(arr)
    # copula = GumbelCopula(theta=theta, k_dim=n_lags)
    # marginals = [bimodal, bimodal]
    # joint_dist = CopulaDistribution(copula, marginals)
    # n = 50
    # y, x = np.meshgrid(np.linspace(data.min(), data.max(), n), np.linspace(data.min(), data.max(), n))
    # arr = np.vstack((y.flatten(), x.flatten())).T
    # z = (1-joint_dist.cdf(arr)).reshape(x.shape)
    #
    # im = plt.pcolormesh(x, y, z, cmap='viridis')
    # plt.colorbar(im)

    # n_lags = 3
    # arr = np.vstack([x[i:-(n_lags-i)] for i in np.arange(n_lags)]).T
    # theta = GumbelCopula(k_dim=n_lags).fit_corr_param(arr)
    # copula = GumbelCopula(theta=theta, k_dim=n_lags)
    # marginals = [bimodal for i in range(n_lags)]
    # joint_dist = CopulaDistribution(copula, marginals)
    # n = 20
    # x = np.linspace(data.min(), data.max(), n).flatten()
    # arr = np.array(list(itertools.product(x, repeat=3)))
    # prob = (1 - joint_dist.cdf(arr))
    #
    # ax = fig.add_subplot(projection='3d')
    # im = ax.scatter(arr[:,0], arr[:,1], arr[:,2], c=prob, cmap='jet', s=60)
    # plt.colorbar(im)
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.title('$p(T_{t} \geq x, T_{t-1} \geq y, T_{t-3} \geq z)$')

    # thresholds = np.arange(25,31)
    thresholds = np.arange(-30,-24)
    lags = np.arange(1,30)
    res = pd.DataFrame(index=lags, columns=thresholds)

    dt = lags.max()
    marginals = [bimodal for i in range(dt)]
    # marg_th_probs = np.array([bimodal.cdf(th)[0] for th in thresholds])
    arr = np.vstack([x[i:-(dt-i)] for i in np.arange(dt)]).T

    theta = GumbelCopula(k_dim=dt).fit_corr_param(arr)
    # copula = GumbelCopula(theta=theta, k_dim=dt)
    # joint_dist = CopulaDistribution(copula, marginals)

    for dt in lags:
        # theta = GumbelCopula(k_dim=dt).fit_corr_param(arr[:,:dt])
        # tmp = np.full(res.T.shape, data.min())
        # tmp[:,:dt] = np.repeat(thresholds.reshape(len(thresholds), 1), dt, axis=1)

        copula = GumbelCopula(theta=theta, k_dim=dt)
        joint_dist = CopulaDistribution(copula, marginals[:dt])
        tmp = np.repeat(thresholds.reshape(len(thresholds), 1), dt, axis=1)
        res.loc[dt, :] = joint_dist.cdf(tmp)

    res.columns = res.columns.values * -1
    res.plot(ax=plt.gca())

    plt.xlabel("Consecutive days with SST above threshold (N)")
    plt.title("$p(SST_t \geq th, SST_{t-1} \geq th, ..., SST_{t-N} \geq th)$")
    plt.legend(title='$th$')

    plt.tight_layout()
    plt.show()

    # print(joint_dist.cdf(([23,24], [23,24])))

class copulas(object):

    def __init__(self, Ser, lags=10):

        self.lags = lags
        self.xSer = Ser * -1
        # detrend and remove seasonality
        x = np.arange(len(self.xSer))
        slope, intercept = np.polyfit(x, self.xSer, 1)
        detrended = pd.Series(self.xSer.values.flatten() - (slope * x), index=self.xSer.index)
        self.clim = calc_anom(detrended, return_clim366=True)

        self.copulas = []

        # climatology for the prior, original data for the evidence
        for data in [self.clim, self.xSer]:

            # Estimate marginal PDF of the data
            gmm = GaussianMixture(n_components=2)
            gmm.fit(data.values.reshape(-1, 1))
            means = gmm.means_
            standard_deviations = gmm.covariances_ ** 0.5
            weights = gmm.weights_
            args = [item for t in zip(means.flatten(), standard_deviations.flatten(), weights) for item in t]
            bimodal = bimodal_gaussian(*args)
            marginals = [bimodal for _ in range(lags)]

            # Construct copulas
            x = data.values.flatten()
            arr = np.vstack([x[i:-(lags - i)] for i in range(lags)]).T
            theta = GumbelCopula(k_dim=lags).fit_corr_param(arr)
            self.copulas += [CopulaDistribution(GumbelCopula(theta=theta, k_dim=lags), marginals)]

    def calc_prior(self, x):

        if len(x) != self.lags:
            print ('calc_prior: wrong input length!')
            return None

        return self.copulas[0].cdf(x*(-1))

    def calc_evidence(self, x):

        if len(x) != self.lags:
            print ('calc_evidence: wrong input length!')
            return None

        return self.copulas[1].cdf(x*(-1))


from scipy.stats import multivariate_normal
def likelyhood(*args):

    t_meas = np.full(4, 30)

    t_true = np.array(args).flatten()

    meas_std = np.array([0.4, 0.4, 0.4, 0.4])
    meas_corr = np.array([
        [1.0, 0.8, 0.6, 0.4],
        [0.8, 1.0, 0.8, 0.6],
        [0.6, 0.8, 1.0, 0.8],
        [0.4, 0.6, 0.8, 1.0],
    ])
    meas_cov = meas_std * meas_corr * meas_std.T

    joint = multivariate_normal(t_true, meas_cov)

    return joint.pdf(t_meas)

def monte_carlo_integration(func, bounds, num_samples=10000):
    samples = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_samples, bounds.shape[0]))
    values = np.apply_along_axis(func, 1, samples)
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    return volume * np.mean(values)

from scipy.integrate import nquad
def calc_posterior():

    # data = pd.read_csv(r"D:\data_sets\SST\GBR\gbr_sst_avg.csv", index_col=0, parse_dates=True)
    #
    # lags = 4
    # threshold = 29
    # cops = copulas(data, lags)
    #
    # print(cops.calc_prior(np.full(lags, threshold)))
    # print(cops.calc_evidence(np.full(lags, threshold)))

    threshold = 27

    ranges = np.array([[threshold, np.inf] for _ in range(4)])

    print(nquad(likelyhood, ranges, opts={'epsabs': 1e-1, 'epsrel': 1e-1}))
    # print(monte_carlo_integration(likelyhood, ranges))

    # meas_std = np.array([1.4, 1.4, 1.4, 1.4])
    # meas_corr = np.array([
    #     [1.0, 0.8, 0.6, 0.4],
    #     [0.8, 1.0, 0.8, 0.6],
    #     [0.6, 0.8, 1.0, 0.8],
    #     [0.4, 0.6, 0.8, 1.0],
    # ])
    # meas_cov = meas_std * meas_corr * meas_std.T
    #
    # # t_true = np.array([26, 26, 26, 26])
    # t_true = np.full(4, 27)
    #
    # t_meas = np.full(4, 30)
    #
    # joint = multivariate_normal(t_true, meas_cov)
    #
    # print(joint.cdf(t_meas))



def Bayes_filter(data, clim, sigma_m=0.1, sigma_t=None):

    if not sigma_t:
        # dtemp = np.diff(data)
        dtemp = np.diff(data-clim)
        sigma_t = dtemp.std()

    dclim = np.diff(clim)

    n = data.size
    # y = np.r_[data / sigma_m, np.zeros(n)]
    y = np.r_[data / sigma_m, 0, dclim / sigma_t]
    A1 = eye_array(n) / sigma_m  # standardize
    A2 = diags_array([np.r_[0, np.ones(n - 1)], -np.ones(n - 1)], offsets=[0, -1]) / sigma_t  # standardize
    A = vstack([A1, A2])

    C = inv(A.T @ A)
    x = C @ (A.T @ y)

    return x, C

def above_threshold_probability(start, count, threshold, x, C):
    m_ = x[start:start+count]
    C_ = C[start:start+count, start:start+count].todense()
    # get probability p(t_i >= T, t_{i+1} >+ T, ... | M) from posterior CDF
    N =  multivariate_normal(m_, C_)
    # this controls the numerical accuracy (default values are slow....)
    N.maxpts = 10000
    N.abseps = 1e-4
    N.releps = 1e-4
    return N.cdf(np.ones(count)*np.inf, lower_limit=np.ones(count)*threshold)


def check_threshold_with_tolerance(arr, threshold, tolerance=3):
    """
    Check if all values in the array exceed a given threshold, allowing up to
    <tolerance> consecutive values below the threshold.

    Parameters:
    arr (numpy.ndarray): The input array of values.
    threshold (float): The threshold value.
    tolerance (int): The maximum number of consecutive values allowed below the threshold (default is 3).

    Returns:
    bool: True if all values exceed the threshold with the tolerance allowed, False otherwise.
    """
    consecutive_below = 0  # Counter for consecutive days below threshold

    for value in arr:
        if value < threshold:
            consecutive_below += 1
            if consecutive_below > tolerance:
                return False  # More than the allowed consecutive days below threshold
        else:
            consecutive_below = 0  # Reset counter if we are above the threshold

    return True  # All values checked within tolerance
def get_above_threshold_prob(x, C, threshold, count=35):

    n = len(x)
    ix_ = []
    pct_above = []  # actual probability calculated considering uncertianties
    all_above = []  # deterministic "benchmark" that <count> actual measurements are above <thresholds>, regardless of uncertainty
    increment = 1
    for i in np.arange(0, n, increment)[:-count]:
        ix_.append(i)
        pct_above.append(
            above_threshold_probability(i, count, threshold, x, C))
        # all_above.append(np.all(x[i:i + count] >= threshold))
        all_above.append(check_threshold_with_tolerance(x[i:i + count], threshold, tolerance=3))
        # print(f" {i / n * 100:.2f} %", end='\r')
    # print(f"100 %")
    ix_ = np.array(ix_)
    pct_above = np.array(pct_above)
    all_above = np.array(all_above)

    return ix_, pct_above, all_above

def plot_above_threshold_prob():
    # 511    3547    3697
    loc = 3547

    sst = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")
    data = pd.DataFrame(sst['sst'].sel(loc=loc).to_pandas())
    data.columns = ['sst']

    # data = pd.read_csv(r"D:\data_sets\SST\GBR\gbr_sst_avg.csv", index_col=0)
    # data.index = pd.DatetimeIndex(data.index)
    # data = data.loc['2010-01-01'::, :]

    clim = calc_anom(data.loc['1981-01-01':'2010-12-31'], mode='climatological', return_clim=True).to_numpy()


    temp = data.loc['2010-01-01'::,'sst'].to_numpy()
    ix = np.arange(temp.size)

    x, C = Bayes_filter(temp, clim, sigma_m=0.1, sigma_t=None)

    # th = np.percentile(x, 90)
    th = data.resample('1M').mean().resample('1Y').max().mean().values[0] + 1 # MMM + 1 (suggested by CM)

    ix_, pct_above, all_above = get_above_threshold_prob(x, C, th)

    plt.figure(figsize=(16, 6))
    plt.plot(ix, temp, label="Measurement data")
    plt.plot(ix, x, label="Smoothed data")
    plt.axhline(th, color='k', linestyle='--', linewidth=1)
    plt.legend(bbox_to_anchor=(0.25, 1.01), loc=8)
    ax = plt.gca().twinx()
    # plotted values represent the probability that this particular day + the following <count> days exceed <threshold> degrees!
    plt.plot(ix_, pct_above*100, color='red', linewidth=1, label="P(above threshold for 30 samples)")
    plt.plot(ix_, all_above*100, color='blue', linewidth=1, label="all above threshold for 30 samples")
    plt.legend(bbox_to_anchor=(0.75, 1.01), loc=8)

    plt.tight_layout()
    plt.show()


import time

def global_analysis(nprocs=None):

    if not nprocs:
        nprocs = os.cpu_count()-6

    if nprocs == 1:
        calc_threshold_probabilities(nprocs, nprocs)
    else:
        p = Pool(nprocs)
        part = np.arange(nprocs) + 1
        parts = repeat(nprocs, nprocs)
        p.starmap(calc_threshold_probabilities, zip(part, parts))


def calc_threshold_probabilities(part, parts):

    dir_out = Path(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted")

    if not dir_out.exists():
        Path.mkdir(dir_out)

    result_file_pct = dir_out / f'result_pct_part{part}.csv'
    result_file_all = dir_out / f'result_all_part{part}.csv'

    # threshold = 29.
    count = 35

    data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")

    n_loc = len(data['loc'])

    subs = (np.arange(parts + 1) * n_loc / parts).astype('int')
    subs[-1] = n_loc
    start = subs[part - 1]
    end = subs[part]-1

    if result_file_pct.exists():
        start = pd.read_csv(result_file_pct, index_col=0).index.values[-1] + 1

    data = data.sel(loc=slice(start, end))

    for i, loc in enumerate(data['loc'].data):

        try:
            print(f'%i / %i' % (i, (end-start)))
            t = time.time()

            # data = data.sel(date=slice('2000-01-01', None))
            tmp_data = data['sst'].sel(loc=loc, date=slice('1981-01-01', '2010-12-31')).to_pandas()
            th = tmp_data.resample('1M').mean().resample('1Y').max().mean() + 1 # MMM + 1

            # tmp_data = data['sst'].sel(loc=loc).data # For absolute values
            tmp_data = data['sst'].sel(loc=loc, date=slice('2000-01-01', None)).to_pandas()
            tmp_unc = data['sst_uncertainty'].sel(loc=loc, date=slice('2000-01-01', None)).data

            tmp_clim = calc_anom(tmp_data, mode='climatological', return_clim=True)

            # th = np.percentile(tmp_data, 90)

            x, C = Bayes_filter(tmp_data.to_numpy(), tmp_clim.to_numpy(), sigma_m=tmp_unc)
            ix_, pct_above, all_above = get_above_threshold_prob(x, C, th, count=count)

            print((time.time() - t))

            res_pct = pd.DataFrame(pct_above.reshape(1,-1), index=(loc,), columns=tmp_data.index.values[:-count])
            res_all = pd.DataFrame(all_above.reshape(1,-1), index=(loc,), columns=tmp_data.index.values[:-count])

            if not result_file_pct.exists():
                res_pct.to_csv(result_file_pct, float_format='%0.4f')
            else:
                res_pct.to_csv(result_file_pct, float_format='%0.4f', mode='a', header=False)

            if not result_file_all.exists():
                res_all.to_csv(result_file_all, float_format='%0.4f')
            else:
                res_all.to_csv(result_file_all, float_format='%0.4f', mode='a', header=False)

            print((time.time() - t))
        except:
            print(f'%i failed' % i)
            continue

    # res.to_netcdf(fout)

def create_lut():

    mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    idx = np.where(mask.data)
    lons, lats = np.meshgrid(mask.lon.values, mask.lat.values)
    lut = pd.DataFrame({'lon': lons[idx],
                        'lat': lats[idx],
                        'row': idx[0],
                        'col': idx[1]})
    lut.to_csv(r"D:\data_sets\SST\GBR\lut.csv")

def plot_sst_map():

    # data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")

    fout = r'H:\work\experiments\coral_bleaching\bleaching_event_prob_map_MMM_35d_det_interrupted.png'

    data = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_pct.csv", index_col=0).T
    data.index = pd.DatetimeIndex(data.index) - pd.Timedelta('183d')

    data_all = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_all.csv", index_col=0).T
    data_all.index = pd.DatetimeIndex(data_all.index) - pd.Timedelta('183d')

    yrs = np.unique(data.index.year)
    yr_label = [f"{yr} / {yr + 1}" for yr in yrs]

    data = data.resample('1YE').max()
    data_all = data_all.resample('1YE').max()

    mask = xr.load_dataarray(r"D:\data_sets\SST\GBR\mask.nc")
    lut = pd.read_csv(r"D:\data_sets\SST\GBR\lut.csv", index_col=0)

    # kick out values that don't exist in 'data'
    lut = lut.loc[data.columns.values]
    lat_north = lut.lat.max() + 1
    lat_south = lut.lat.min() - 1
    lon_west = lut.lon.min() - 1
    lon_east = lut.lon.max() + 1
    img_extent = [lon_west, lon_east, lat_south, lat_north]

    lons, lats = np.meshgrid(mask.lon.values, mask.lat.values)

    # arr = data['sst'].sel(date='2024-03-01 12:00:00')
    # arr = data.loc['2019-03-01 12:00:00',:]
    arr1 = data.loc['2019-12-31',:]
    arr2 = data.loc['2023-12-31',:]
    arr3 = data_all.loc['2019-12-31',:]
    arr4 = data_all.loc['2023-12-31',:]
    titles = ['2019 / 2020 (prob.)', '2023 / 2024 (prob.)', '2019 / 2020 (det.)', '2023 / 2024 (det.)']

    crs = ccrs.PlateCarree()
    f = plt.figure(figsize=(14, 16))
    axs = []
    ims = []

    # cm = colors.ListedColormap(sns.color_palette('plasma', n_colors=2))
    cm = colors.ListedColormap([colormaps['plasma'](0)[0:3], colormaps['plasma_r'](0)[0:3]])

    for i, (arr, title) in enumerate(zip([arr1,arr2,arr3,arr4],titles)):

        img = np.full(mask.shape, np.nan)
        img[lut.row, lut.col] = arr

        # Plot the data using a PlateCarree projection
        ax = f.add_subplot(2, 2, i+1, projection=crs)
        ax.set_extent(img_extent)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        ims += [ax.pcolormesh(lons, lats, img, transform=crs, cmap='plasma' if i<2 else cm, vmin=0.0, vmax=0.5)]

        fname = r"D:\data_sets\SST\GBR\worldheritagemarineprogramme.shp"
        shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                        ccrs.PlateCarree(), facecolor='none')
        ax.add_feature(shape_feature, linewidth=1, edgecolor='firebrick')
        ax.set_title(title)

        axs += [ax]

    f.suptitle('Maximum probability of conditions that favor coral bleaching', y=0.94)

    plt.subplots_adjust(wspace=0.01, hspace=0.09)
    f.subplots_adjust(right=0.91)
    # cbar_ax = f.add_axes([0.87, 0.25, 0.02, 0.5])

    # pos2 = axs[1].get_position()
    # pos3 = axs[2].get_position()
    #
    # cbar_height = pos2.y1 - pos3.y0  # Matches the height of the first plot
    # cbar_bottom = pos3.y0  # Aligns with the bottom of the first plot
    #
    # cbar_ax = f.add_axes([pos2.x1 + 0.02, cbar_bottom, 0.024, cbar_height])  # [left, bottom, width, height]

    pos1 = axs[0].get_position()
    pos2 = axs[1].get_position()
    pos3 = axs[2].get_position()
    pos4 = axs[3].get_position()

    top_cbar_height = pos1.y1 - pos1.y0
    bottom_cbar_height = pos3.y1 - pos3.y0
    top_cbar_bottom = pos1.y0
    bottom_cbar_bottom = pos3.y0

    top_cbar_ax = f.add_axes([pos2.x1 + 0.02, top_cbar_bottom, 0.024, top_cbar_height])
    bottom_cbar_ax = f.add_axes([pos4.x1 + 0.02, bottom_cbar_bottom, 0.024, bottom_cbar_height])

    f.colorbar(ims[0], cax=top_cbar_ax)
    cb = f.colorbar(ims[3], cax=bottom_cbar_ax)
    cb.set_ticks([0.125, 0.375])
    cb.ax.set_yticklabels(["No", "Yes"])

    f.savefig(fout, dpi=200, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def plot_n_bleaching_events_ts():

    fout = r'H:\work\experiments\coral_bleaching\bleaching_event_frequency_MMM_35d_det_interrupted.png'
    data = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_pct.csv", index_col=0).T
    data.index = pd.DatetimeIndex(data.index) - pd.Timedelta('183d')
    data = data.loc['2000-01-01'::, :]

    data_all = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_all.csv", index_col=0).T
    data_all.index = pd.DatetimeIndex(data_all.index) - pd.Timedelta('183d')
    data_all = data_all.loc['2000-01-01'::, :]

    yrs = np.unique(data.index.year)
    yr_label = [f"{yr} / {yr + 1}" for yr in yrs]

    data = data.resample('1YE').max()
    data_all = data_all.resample('1YE').max()
    n = len(data.columns)

    res = pd.DataFrame({'>10 %': (data.values>=0.10).sum(axis=1) / n * 100,
                        '>50 %': (data.values>=0.50).sum(axis=1) / n * 100,
                        # '>95 %': (data.values>=0.95).sum(axis=1) / n * 100,
                        'det.': (data_all.values==1).sum(axis=1) / n * 100}, index=yr_label)

    f, ax = plt.subplots(1,1,figsize=(15,5))
    [plt.axvline(x, linewidth=0.2,linestyle=':', color='black') for x in np.arange(len(yr_label))]
    # [plt.axvline(x, linewidth=1,linestyle='--', color='firebrick') for x in [1,2,19,20]]

    # res.plot(ax=ax, marker='o', linestyle='--', linewidth=1, markersize=10)
    markers = ['o','o','^']
    for i, col in enumerate(res.columns):
        res.plot(y=col, marker=markers[i], ax=ax, label=col, linestyle='--', linewidth=1, markersize=8)

    plt.title('Fraction of gridcells [%] exhibiting coral bleaching conditions at different confidence levels')

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    f.savefig(fout, dpi=200, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def plot_sst_unc_prob_ts():

    sst = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")

    start = '2020-01-15'
    end = '2020-03-15'

    prob = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_pct.csv", index_col=0).T
    prob.index = pd.DatetimeIndex(prob.index)
    prob = prob.loc[start:end, :]

    det = pd.read_csv(r"D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted_th_clim\result_all.csv",
                           index_col=0).T
    det.index = pd.DatetimeIndex(det.index)
    det = det.loc[start:end, :]

    loc1 = prob.columns[np.where((prob>0.1)&(prob<0.11))[1][0]]
    loc2 = prob.columns[np.where((prob>0.35)&(prob<0.37))[1][0]]
    # loc3 = prob.columns[np.where((prob>0.60)&(prob<0.70))[1][0]]
    loc3 = prob.columns[np.where(prob==prob.max().max())[1][0]]

    print(loc1, loc2, loc3)

    # ts1 = sst.sel(date=slice(start,end), loc=loc1).to_pandas().drop('loc',axis='columns')
    # ts2 = sst.sel(date=slice(start,end), loc=loc2).to_pandas().drop('loc',axis='columns')
    # ts3 = sst.sel(date=slice(start,end), loc=loc3).to_pandas().drop('loc',axis='columns')
    ts1 = sst.sel(loc=loc1).to_pandas().drop('loc',axis='columns')
    ts2 = sst.sel(loc=loc2).to_pandas().drop('loc',axis='columns')
    ts3 = sst.sel(loc=loc3).to_pandas().drop('loc',axis='columns')

    tss = [ts1, ts2, ts3]
    probs = [prob[loc1], prob[loc2], prob[loc3]]
    dets = [det[loc1], det[loc2], det[loc3]]

    fig, axs = plt.subplots(3,1,figsize=(16, 12),sharex=True)
    for i,(ax, ts, p, d) in enumerate(zip(axs, tss, probs, dets)):

        th = ts.loc['1981-01-01':'2010-12-31','sst'].resample('1M').mean().resample('1Y').max().mean() + 1
        ts = ts.loc[start:end]

        ax.plot(ts.index, ts['sst'], label='SST', color='blue')

        # Plot the shaded area for standard deviation

        ax.fill_between(ts.index,
                         ts['sst'] - 2*ts['sst_uncertainty'],
                         ts['sst'] + 2*ts['sst_uncertainty'],
                         color='purple', alpha=0.2, label='2-sigma')
        ax.fill_between(ts.index,
                         ts['sst'] - ts['sst_uncertainty'],
                         ts['sst'] + ts['sst_uncertainty'],
                         color='blue', alpha=0.2, label='1-sigma')

        # th = np.percentile(ts['sst'], 90)
        ax.axhline(th, linestyle='--', color='k', linewidth=1, label='MMM+1')
        ax.set_ylim([th-2.5, th+2.5])

        ax2 = ax.twinx()
        ax2.plot(p.index, p.values, label='p(SST > MMM+1)', color='green')
        ax2.fill_betweenx([-0.1, 1.1],
                         ts[ts['sst']>=th].index[0],
                         ts[ts['sst']>=th].index[-1],
                         color='green', alpha=0.15, label='SST$_{det}$ > MMM+1')
        # ax2.plot(d.index, d.values, label='det.', color='red')
        ax2.set_ylim([-0.1, 1.1])

        if i==0:
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        if i==1:
            ax.set_ylabel('SST [$^\circ$]')
            ax2.set_ylabel('Probability [-]')
    # plt.legend()

    plt.tight_layout()
    plt.show()


def test_random_walk():

    data = pd.read_csv(r"D:\data_sets\SST\GBR\gbr_sst_avg.csv", index_col=0, parse_dates=True)

    anom = calc_anom(data, mode='climatological')

    temp = data['sst'].to_numpy()
    temp_anom = anom.to_numpy()
    ix = np.arange(temp.size)

    dtemp = np.diff(temp)
    dtemp_anom = np.diff(temp_anom)

    # plt.figure(figsize=(12, 3))
    # plt.plot(ix, temp)
    # plt.show()
    #
    # plt.figure(figsize=(12, 3))
    # plt.plot(ix[1:], dtemp)
    # plt.show()

    f, ax = plt.subplots(1,2, figsize=(16,8))
    sns.histplot(dtemp, kde=True, ax=ax[0])
    sns.histplot(dtemp_anom, kde=True, ax=ax[1])
    print(f"mean={dtemp.mean():.6f}, stdev={dtemp.std():.6f}")
    print(f"mean={dtemp_anom.mean():.6f}, stdev={dtemp_anom.std():.6f}")
    print((dtemp.std() - dtemp_anom.std())/dtemp.std())

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    # plot_avg_ts()
    # plot_bleaching_events()
    # extract_gbr_sst_unc()
    # fit_bimodal()
    # calc_posterior()

    # create_lut()

    # plot_above_threshold_prob()

    # calc_threshold_probabilities(1, 1)
    # global_analysis()

    # test_random_walk()
    #
    plot_sst_unc_prob_ts()
    # plot_n_bleaching_events_ts()
    # plot_sst_map()

    # from myprojects.functions import merge_files
    # merge_files(r'D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted', pattern='result_all*.csv', fname='result_all.csv', delete=False)
    # merge_files(r'D:\data_sets\SST\GBR\sst_gbr_threshold_probs_MMA_det_interrupted', pattern='result_pct*.csv', fname='result_pct.csv', delete=False)

    # data = xr.load_dataset(r"D:\data_sets\SST\GBR\sst_gbr.nc")
    # data = data['sst'].mean(dim='loc').to_series()
    # data.to_csv(r'D:\data_sets\SST\GBR\gbr_sst_avg.csv')
