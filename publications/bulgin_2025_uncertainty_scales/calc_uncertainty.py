import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set_context('talk', font_scale=0.8)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from myprojects.io.ascat import HSAF_io
from myprojects.io.smap import SMAP_io
from myprojects.io.gldas import GLDAS_io

from validation_good_practice.ancillary.metrics import TCA_calc

class ioo(object):

    def __init__(self):
        self.ascat = HSAF_io()
        self.smap = SMAP_io()
        self.gldas = GLDAS_io()
        self.lut = pd.read_csv(r"D:\data_sets\lut_smap_ascat_amsr2.csv", index_col=0)

    def read(self, loc):
        try:
            ts_ascat = self.ascat.read(loc['gpi_asc']).resample('1D').mean()
            ts_smap = self.smap.read(loc['ease_row'], loc['ease_col'],rowcol=True).resample('1D').mean() * 100
            ts_gldas = self.gldas.read(int(loc['gpi_ams']))
            ts_gldas = ts_gldas[ts_gldas.index.hour == 0]
            df = pd.concat((ts_ascat,ts_smap,ts_gldas),axis=1).dropna()
        except:
            df = None

        return df

def calc_spat_err(io, loc, n_lags=5):

    df = io.read(loc)

    if len(df) < 100:
        raise ValueError("Not enough samples!")

    _, err_std, beta = TCA_calc(df)
    err_std /= beta  # error standard deviation

    n = len(df)
    r = df.corr().values

    err_corr = pd.DataFrame(columns=df.columns, index=np.arange(n_lags))
    err_corr.iloc[0,:] = [1, 1, 1]

    for lag in np.arange(1, n_lags):

        loc_n = io.lut[(io.lut['ease_row'] == (loc['ease_row'] - lag)) & (io.lut['ease_col'] == loc['ease_col'])].iloc[0]
        loc_s = io.lut[(io.lut['ease_row'] == (loc['ease_row'] + lag)) & (io.lut['ease_col'] == loc['ease_col'])].iloc[0]
        loc_w = io.lut[(io.lut['ease_col'] == (loc['ease_col'] - lag)) & (io.lut['ease_row'] == loc['ease_row'])].iloc[0]
        loc_e = io.lut[(io.lut['ease_col'] == (loc['ease_col'] + lag)) & (io.lut['ease_row'] == loc['ease_row'])].iloc[0]

        tmp_res = pd.DataFrame(columns=df.columns, index=np.arange(4))
        for i, tmp_loc in enumerate([loc_n, loc_s, loc_w, loc_e]):

            dfl = io.read(tmp_loc)

            _, err_std_l, beta_l = TCA_calc(dfl)
            err_std_l /= beta_l

            tmp_res.iloc[i,0] = (df.iloc[:,0].cov(dfl.iloc[:,0]) - df.iloc[:,0].cov(dfl.iloc[:,1]) * df.iloc[:,2].cov(dfl.iloc[:,0]) / df.iloc[:,2].cov(dfl.iloc[:,1])) / (err_std[0] * err_std_l[0])
            tmp_res.iloc[i,1] = (df.iloc[:,1].cov(dfl.iloc[:,1]) - df.iloc[:,1].cov(dfl.iloc[:,0]) * df.iloc[:,2].cov(dfl.iloc[:,1]) / df.iloc[:,2].cov(dfl.iloc[:,0])) / (err_std[1] * err_std_l[1])
            tmp_res.iloc[i,2] = (df.iloc[:,2].cov(dfl.iloc[:,2]) - df.iloc[:,2].cov(dfl.iloc[:,0]) * df.iloc[:,1].cov(dfl.iloc[:,2]) / df.iloc[:,1].cov(dfl.iloc[:,0])) / (err_std[2] * err_std_l[2])

        err_corr.iloc[lag,:] = tmp_res.mean(axis=0).values

    err_corr[(err_corr>1) | (err_corr<-1)] = np.nan

    return err_std**2, err_corr, r, n
def calc_temp_err(df, n_lags=31):

    _, err, beta = TCA_calc(df)
    err /= beta  # error standard deviation
    err **= 2

    dfl = df.copy()

    res = pd.DataFrame(columns=df.columns, index=np.arange(n_lags))
    res.iloc[0,:] = [1, 1, 1]

    for i in np.arange(1, n_lags):
        dfl.index -= pd.Timedelta('1D')
        res.iloc[i,0] = (df.iloc[:,0].cov(dfl.iloc[:,0]) - df.iloc[:,0].cov(dfl.iloc[:,1]) * df.iloc[:,2].cov(dfl.iloc[:,0]) / df.iloc[:,2].cov(dfl.iloc[:,1])) / err[0]
        res.iloc[i,1] = (df.iloc[:,1].cov(dfl.iloc[:,1]) - df.iloc[:,1].cov(dfl.iloc[:,0]) * df.iloc[:,2].cov(dfl.iloc[:,1]) / df.iloc[:,2].cov(dfl.iloc[:,0])) / err[1]
        res.iloc[i,2] = (df.iloc[:,2].cov(dfl.iloc[:,2]) - df.iloc[:,2].cov(dfl.iloc[:,0]) * df.iloc[:,1].cov(dfl.iloc[:,2]) / df.iloc[:,1].cov(dfl.iloc[:,0])) / err[2]

    res[(res>1) | (res<-1)] = np.nan

    return err, res

def calc_spatial_error_correlations():

    io = ioo()

    lut = pd.read_csv(r"D:\data_sets\lut_smap_ascat_amsr2.csv", index_col=0)

    # latmin, latmax = 37, 44
    # lonmin, lonmax = -110, -90
    latmin, latmax = 26, 49
    lonmin, lonmax = -125, -67
    lut = lut[(lut.lat>=latmin)&(lut.lat<=latmax)&(lut.lon>=lonmin)&(lut.lon<=lonmax)]
    lut.index = np.arange(len(lut))

    # lat, lon = 38.791223158901815, -95.70362512903604
    # ind = np.argmin((lut.lat - lat)**2 + (lut.lon - lon)**2)
    # lut = lut.loc[ind:ind+3]
    # lut.index = np.arange(len(lut))

    sensors = ['ascat','smap','gldas']
    lags = np.arange(5)
    res = xr.Dataset(
        data_vars=dict(
            err_var=(["loc", "sensor"], np.full((len(lut),len(sensors)), np.nan)),
            err_corr=(["loc", "lag", "sensor"], np.full((len(lut),len(lags),len(sensors)), np.nan)),
            r=(["loc", "sensor1", "sensor2"], np.full((len(lut),len(sensors),len(sensors)), np.nan)),
            p=(["loc", "sensor1", "sensor2"], np.full((len(lut),len(sensors),len(sensors)), np.nan)),
            n=(["loc",], np.full((len(lut)), np.nan)),),
        coords=dict(
            lon=("loc", lut['lon'].values),
            lat=("loc", lut['lat'].values),
            col=("loc", lut['ease_col'].values),
            row=("loc", lut['ease_row'].values),
            sensor=sensors,
            sensor1=sensors,
            sensor2=sensors,
            lag=lags,
        ),)

    for idx, loc in lut.iterrows():

        print(f'{idx+1} / {len(lut)}')

        try:
            err_var, err_corr, r, n = calc_spat_err(io, loc, n_lags=len(lags))
        except:
            continue

        res['r'].loc[dict(loc=idx)] = r
        res['n'].loc[dict(loc=idx)] = n
        res['err_var'].loc[dict(loc=idx)] = err_var
        res['err_corr'].loc[dict(loc=idx)] = err_corr

    res.to_netcdf(r'H:\work\experiments\uncertainty_scales\spat_corr.nc')

def calc_temp_error_correlations():

    ascat = HSAF_io()
    smap = SMAP_io()
    gldas = GLDAS_io()

    lut = pd.read_csv(r"D:\data_sets\lut_smap_ascat_amsr2.csv", index_col=0)

    # lat, lon = 38.791223158901815, -95.70362512903604
    # ind = np.argmin((lut.lat - lat)**2 + (lut.lon - lon)**2)
    # lut = lut.loc[ind:ind+3]
    # lut.index = np.arange(len(lut))

    latmin, latmax = 26, 49
    lonmin, lonmax = -125, -67
    lut = lut[(lut.lat>=latmin)&(lut.lat<=latmax)&(lut.lon>=lonmin)&(lut.lon<=lonmax)]
    lut.index = np.arange(len(lut))

    sensors = ['ascat','smap','gldas']
    lags = np.arange(31)
    res = xr.Dataset(
        data_vars=dict(
            err_var=(["loc", "sensor"], np.full((len(lut),len(sensors)), np.nan)),
            err_corr=(["loc", "lag", "sensor"], np.full((len(lut),len(lags),len(sensors)), np.nan)),
            r=(["loc", "sensor1", "sensor2"], np.full((len(lut),len(sensors),len(sensors)), np.nan)),
            p=(["loc", "sensor1", "sensor2"], np.full((len(lut),len(sensors),len(sensors)), np.nan)),
            n=(["loc",], np.full((len(lut)), np.nan)),),
        coords=dict(
            lon=("loc", lut['lon'].values),
            lat=("loc", lut['lat'].values),
            col=("loc", lut['ease_col'].values),
            row=("loc", lut['ease_row'].values),
            sensor=sensors,
            sensor1=sensors,
            sensor2=sensors,
            lag=lags,
        ),)

    for idx, loc in lut.iterrows():

        print(f'{idx+1} / {len(lut)}')

        try:
            ts_ascat = ascat.read(loc['gpi_asc']).resample('1D').mean()
            ts_smap = smap.read(loc['ease_row'], loc['ease_col'],rowcol=True).resample('1D').mean() * 100
            ts_gldas = gldas.read(int(loc['gpi_ams']))
            ts_gldas = ts_gldas[ts_gldas.index.hour == 0]
            df = pd.concat((ts_ascat,ts_smap,ts_gldas),axis=1).dropna()
        except:
            continue

        if len(df) < 100:
            continue

        res['n'].loc[dict(loc=idx)] = len(df)
        res['r'].loc[dict(loc=idx)] = df.corr().values
        err, corr = calc_temp_err(df, n_lags=len(lags))
        res['err_var'].loc[dict(loc=idx)] = err
        res['err_corr'].loc[dict(loc=idx)] = corr


    res.to_netcdf(r'H:\work\experiments\uncertainty_scales\temp_corr.nc')

def plot_err_corr():

    fout = r"H:\work\experiments\uncertainty_scales\error_autocorrelation.png"

    spat_corr = xr.open_dataset(r"H:\work\experiments\uncertainty_scales\spat_corr.nc")
    temp_corr = xr.open_dataset(r"H:\work\experiments\uncertainty_scales\temp_corr.nc")

    spat_corr_avg = spat_corr['err_corr'].mean(dim='loc',skipna=True).to_dataframe().pivot_table(index='lag', columns='sensor')
    temp_corr_avg = temp_corr['err_corr'].mean(dim='loc',skipna=True).to_dataframe().pivot_table(index='lag', columns='sensor')

    spat_corr_avg = spat_corr_avg.loc[:,[('err_corr', 'ascat'),('err_corr', 'smap'),('err_corr', 'gldas')]]
    temp_corr_avg = temp_corr_avg.loc[:,[('err_corr', 'ascat'),('err_corr', 'smap'),('err_corr', 'gldas')]]

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 3])

    ax1 = fig.add_subplot(gs[0, 0])
    temp_corr_avg.plot(ax=ax1)
    ax1.set_title('Temporal error auto-correlation (-)')
    ax1.set_xlabel('lag (days)')
    ax1.legend(np.char.upper(temp_corr['sensor'].values.astype('str')))


    ax2 = fig.add_subplot(gs[0, 1])
    spat_corr_avg.plot(ax=ax2)
    ax2.set_title('Spatial error auto-correlation (-)')
    ax2.set_xlabel('lag (degrees)')
    ax2.set_xticks(np.arange(5), np.arange(5)*0.25)
    ax2.legend(np.char.upper(spat_corr['sensor'].values.astype('str')))

    fig.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def get_err_reduction_temp(v):

    n = len(v)
    o = np.matrix(np.ones(n)) # Ones vector
    m = np.matrix(np.eye(n)) # Error covariance matrix to be filled
    for i in range(n):
        m[i, i::] = v[0:n - i]
        m[i, 0:i] = v[::-1][n - i - 1:-1]

    return (1 - (o * m.I * o.T).I)[0,0]

def get_err_reduction_spat(v):

    n = len(v)
    o = np.matrix(np.ones(n**2)) # Ones vector
    m = np.matrix(np.eye(n**2)) # Error covariance matrix to be filled
    for i in np.arange(n**2):
        for j in np.arange(n**2):
            if i == j:
                continue
            x1, y1 = np.unravel_index(i, (n,n))
            x2, y2 = np.unravel_index(j, (n,n))
            ind = int(np.sqrt((x2-x1)**2+(y2-y1)**2))
            if ind < len(v):
                m[i, j] = v[int(np.sqrt((x2-x1)**2+(y2-y1)**2))]
            else:
                m[i, j] = v[-1]
    # print(m)
    return (1 - (o * m.I * o.T).I)[0,0]

def plot_err_reduction():

    fout = r"H:\work\experiments\uncertainty_scales\soil_moisture_uncertainty_reduction.png"

    spat_corr = xr.open_dataset(r"H:\work\experiments\uncertainty_scales\spat_corr.nc")
    temp_corr = xr.open_dataset(r"H:\work\experiments\uncertainty_scales\temp_corr.nc")

    spat_corr_avg = spat_corr['err_corr'].mean(dim='loc',skipna=True).to_dataframe().pivot_table(index='lag', columns='sensor')
    temp_corr_avg = temp_corr['err_corr'].mean(dim='loc',skipna=True).to_dataframe().pivot_table(index='lag', columns='sensor')

    v0 = np.zeros(len(temp_corr_avg))
    v0[0] = 1
    res_temp = pd.DataFrame(index=np.arange(len(v0)), columns=['ascat','smap','gldas','uncorrelated'],dtype='float')
    res_temp.iloc[0,:] = 1
    for sensor in ['ascat','smap','gldas']:
        v = temp_corr_avg.loc[:,('err_corr', sensor)].values
        res_temp.loc[:, sensor] = [get_err_reduction_temp(v[0:l]) for l in np.arange(1,len(v)+1)]
    res_temp.loc[:, 'uncorrelated'] = [get_err_reduction_temp(v0[0:l]) for l in np.arange(1,len(v0)+1)]

    v0 = np.zeros(len(spat_corr_avg))
    v0[0] = 1
    res_spat = pd.DataFrame(index=np.arange(len(v0)), columns=['ascat','smap','gldas','uncorrelated'],dtype='float')
    res_spat.iloc[0,:] = 1
    for sensor in ['ascat','smap','gldas']:
        v = spat_corr_avg.loc[:,('err_corr', sensor)].values
        res_spat.loc[:, sensor] = [get_err_reduction_spat(v[0:l]) for l in np.arange(1,len(v)+1)]
    res_spat.loc[:, 'uncorrelated'] = [get_err_reduction_spat(v0[0:l]) for l in np.arange(1,len(v0)+1)]


    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 3])

    ax1 = fig.add_subplot(gs[0, 0])
    res_temp.plot(ax=ax1,
               style=['-','-','-','--'])
    ax1.set_title('Uncertainty reduction upon temporal averaging (-)')
    ax1.set_xlabel('Averaging time scale (days)')
    ax1.legend(np.char.upper(res_temp.columns.values.astype('str')))


    ax2 = fig.add_subplot(gs[0, 1])
    res_spat.plot(ax=ax2,
                  style=['-', '-', '-', '--'])
    ax2.set_title('Uncertainty reduction upon spatial averaging (-)')
    ax2.set_xlabel('Averaging length scale (degrees)')
    ax2.set_xticks(np.arange(5), np.arange(5)*0.25)
    ax2.legend(np.char.upper(res_spat.columns.values.astype('str')))

    fig.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

if __name__=='__main__':

    # calc_spatial_error_correlations()
    # calc_temp_error_correlations()

    # plot_err_corr()
    plot_err_reduction()

    # v = np.array([1,0.8,0.6, 0.3])
    # get_err_reduction_spat(v)