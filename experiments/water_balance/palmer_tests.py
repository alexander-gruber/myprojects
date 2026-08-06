import warnings
warnings.filterwarnings("ignore")

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

from myprojects.io.era import E5L_io
from myprojects.io.gleam import GLEAM_io
from myprojects.io.insitu import ISMN_io
from myprojects.io.rooting_depth import RD_io

from myprojects.experiments.palmer_model.palmer import calibrate_Palmer_model, calc_Palmer_SM

def read_Palmer_forcing(lon, lat):

    rd = RD_io()
    era = E5L_io()
    gleam = GLEAM_io(var='E')

    porosity_map = xr.open_dataset(r"D:\data_sets\SMECV\ESA_CCI_SM_v07.1\000_ancillary\ESACCI-SOILMOISTURE-POROSITY_V01.1.nc")
    porosity = float(porosity_map['porosity'].sel(lat=lat, lon=lon, method='nearest').data) / 100.
    if np.isnan(porosity):
        porosity = 0.5
    rooting_depth = rd.read(lon, lat)

    ts_era = era.read(lon, lat)['2010-01-01':'2021-12-31']
    ts_gleam = gleam.read(lon, lat)['2010-01-01':'2021-12-31']
    df = pd.concat((ts_era,ts_gleam),axis=1)

    return df, rooting_depth, porosity

def get_test_ts():

    era = E5L_io()
    gleam = GLEAM_io(var='E')

    # lat, lon = 48.48184246565035, 16.237024018815355 # Austria cropland
    lat, lon = 31.5, -83.55 # Little River

    ts_era = era.read(lon, lat)['2010-01-01':'2021-12-31']
    ts_gleam = gleam.read(lon, lat)['2010-01-01':'2021-12-31']
    df = pd.concat((ts_era,ts_gleam),axis=1)
    df.to_csv(r'H:\work\experiments\palmer_model\test_little_river.txt',float_format='%0.6f')

def plot_palmer_ts():

    lat, lon = 31.5, -83.55
    # df, rooting_depth, porosity = read_Palmer_forcing(lon, lat)
    # print(rooting_depth, porosity)

    # calibration results Little River
    alpha = 1
    gamma = 0.95
    rooting_depth = 450
    porosity = 0.44
    df = pd.read_csv(r'H:\work\experiments\palmer_model\test_little_river.txt', index_col=0, parse_dates=True)

    kwargs = {'alpha': alpha,
              'gamma': gamma,
              'depth_surface': 100,
              'depth_rootzone': rooting_depth,
              'porosity': porosity}
    sm = calc_Palmer_SM(df, **kwargs)

    _, ax = plt.subplots(5, 1, figsize=(15, 10), sharex='all')

    df.plot(ax=ax[0])
    sm[['ssm', 'susm', 'rzsm']].plot(ax=ax[1])
    sm[['f_dSM']].plot(ax=ax[2])
    sm[['dEP']].plot(ax=ax[3])
    sm[['F']].plot(ax=ax[4])

    plt.tight_layout()
    plt.show()

def test_palmer_calibration():

    plot = True

    ismn = ISMN_io()
    network = 'SCAN'
    station = 'LittleRiver'
    ref = ismn.read(network, station).resample('1D').mean().loc['2015-01-01':'2020-12-31',:]
    lat, lon = ismn.get_station_coordinates(network, station)

    # df, rooting_depth, porosity = read_Palmer_forcing(lon, lat)
    df = pd.read_csv(r'H:\work\experiments\palmer_model\test_little_river.txt', index_col=0, parse_dates=True)
    rooting_depth = 450
    porosity = 0.44

    kwargs = {'depth_surface': 100.,
              'depth_rootzone': rooting_depth,
              'porosity': porosity}

    # alpha = 5
    # gamma = 0.95
    alpha, gamma = calibrate_Palmer_model(df, ref['sm_profile'], target='rzsm', **kwargs)

    sm = calc_Palmer_SM(df, alpha=alpha, gamma=gamma, **kwargs)
    print(sm['rzsm'].corr(ref['sm_profile']))
    print(sm['ssm'].corr(ref['sm_surface']))

    # gleam = GLEAM_io(var='SMrz')
    # ts_gleam = gleam.read(lon, lat)['2015-01-01':'2020-12-31']
    # print(ts_gleam.corr(ref['sm_profile']))
    # print(ts_gleam.corr(sm['rzsm']))
    #
    # era = E5L_io(parameter='swvl2')
    # ts_era = era.read_sm(lon, lat)['2015-01-01':'2020-12-31']
    # print(ts_era.corr(ref['sm_profile']))
    # print(ts_era.corr(sm['rzsm']))

    if plot:
        df2 = pd.concat((sm['rzsm'], ref['sm_profile']),axis=1).dropna()
        df2['rzsm'] = (df2['rzsm'] - df2['rzsm'].mean())/df2['rzsm'].std() * df2['sm_profile'].std() + df2['sm_profile'].mean()

        _, ax = plt.subplots(3, 1, figsize=(15, 10), sharex='all')

        ref.plot(ax=ax[0])
        sm[['ssm', 'susm', 'rzsm']].plot(ax=ax[1])
        df2.plot(ax=ax[2])

        # ts_gleam.plot(ax=ax[1])
        # ts_era.plot(ax=ax[1])

        plt.tight_layout()
        plt.show()


if __name__=='__main__':

    # plot_palmer_ts()
    test_palmer_calibration()