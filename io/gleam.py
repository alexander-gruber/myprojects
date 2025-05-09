import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path

class GLEAM_io(object):

    def __init__(self, var='E'):
        path = Path(r'D:\data_sets\GLEAM\v4.2a\daily')
        files = list(path.glob(f'**/{var}_*.nc'))
        self.ds = xr.open_mfdataset(files, chunks='auto')
        self.var = var

    def read(self, lon, lat):

        res = self.ds[f'{self.var}'].sel(lat=lat, lon=lon, method='nearest').to_pandas()
        res.name = 'evaporation'
        return res

