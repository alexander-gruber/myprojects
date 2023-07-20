

import os
import platform

import numpy as np
import pandas as pd

from pathlib import Path

from netCDF4 import Dataset, num2date

from ascat.read_native.cdr import load_grid

from pyldas.interface import LDAS_io

class HSAF_io(object):

    def __init__(self, root=None, version='h119', ext='h120'):

        if root is None:
            # if platform.system() == 'Linux':
            #     self.root = Path('/staging/leuven/stg_00024/OUTPUT/alexg/data_sets/ASCAT')
            # else:
            # self.root = Path(r"R:\Projects\H_SAF_CDOP3\05_deliverables_products")
            self.root = Path(r"D:\data_sets\HSAF")
        else:
            self.root = Path(root)

        self.data_path = os.path.join(self.root, version)
        self.version = version.upper()

        self.grid = load_grid(self.root.parent / 'auxiliary' / 'warp5_grid' / 'TUW_WARP5_grid_info_2_3.nc')
        self.gpis = self.grid.activegpis
        self.cells = self.grid.activearrcell

        self.loaded_cell = None
        self.fid = None

        if ext is not None:
            self.ext = HSAF_io(root=self.root, version=ext, ext=None)
        else:
            self.ext = None

        # self.frozen_snow_prob = xr.open_dataset(os.path.join(self.root, 'static_layer', 'frozen_snow_probability.nc'))
        # quite slow to read!

    def latlon2gpi(self, lat, lon):
        return self.grid.find_nearest_gpi(lon, lat, max_dist=10000)[0]

    def load(self, cell):

        fname = os.path.join(self.data_path, self.version + '_%04i.nc' % cell)
        if not os.path.exists(fname):
            print('File not found: ' + fname)
            return False

        try:
            if self.fid is not None:
                self.fid.close()
            self.fid = Dataset(fname)
        except:
            print('Corrupted cell: %i' % cell)
            return False

        self.loaded_cell = cell

        return True

    def read(self, *args, var='sm', sampling_freq=None, ascat_id=None):

        if len(args) == 1:
            gpi = int(args[0])
        else:
            gpi = self.latlon2gpi(*args)

        if not gpi in self.gpis:
            print('GPI not found')
            return None

        cell = self.cells[self.gpis==gpi][0]

        if self.loaded_cell != cell:
            loaded = self.load(cell)
            if loaded is False:
                return None

        loc = np.where(self.fid['location_id'][:] == gpi)[0][0]
        start = self.fid['row_size'][0:loc].sum()
        end = start + self.fid['row_size'][loc]

        data = self.fid[var][start:end]

        if var != 'sm':
            ind_valid = np.arange(len(data))
        else:
            corr_flag = self.fid['corr_flag'][start:end]
            conf_flag = self.fid['conf_flag'][start:end]
            proc_flag = self.fid['proc_flag'][start:end]
            ssf = self.fid['ssf'][start:end]

            if ascat_id is not None:
                sat_id = self.fid['sat_id'][start:end]
                if ascat_id.upper() == 'A':
                    satid = 3
                elif ascat_id.upper() == 'B':
                    satid = 4
                else:
                    satid = 5
                ind_valid = ((corr_flag == 0) | (corr_flag == 4)) & (conf_flag == 0) & (proc_flag == 0) & (ssf == 1) & (sat_id == satid)
            else:
                ind_valid = ((corr_flag==0)|(corr_flag==4)) & (conf_flag == 0) & (proc_flag == 0) & (ssf == 1)
            if len(np.where(ind_valid)[0]) == 0:
                print('No valid ASCAT soil moisture data for gpi %i' % gpi)
                return None

        if sampling_freq is not None:
            k = 24/sampling_freq
            time = num2date((self.fid['time'][start:end][ind_valid]*k).round()/k, units=self.fid['time'].units,
                            only_use_python_datetimes=True, only_use_cftime_datetimes=False)
        else:
            time = num2date(self.fid['time'][start:end][ind_valid], units=self.fid['time'].units,
                            only_use_python_datetimes=True, only_use_cftime_datetimes=False)

        ts = pd.Series(data[ind_valid], index=time)
        ts = ts[~ts.index.duplicated(keep='first')]

        if self.ext is not None:
            ts_ext = self.ext.read(gpi, var=var, sampling_freq=sampling_freq)
            ts = pd.concat((ts,ts_ext))

        ts.name = 'ascat'
        return ts

    def close(self):
        if self.fid is not None:
            self.fid.close()
        if self.ext is not None:
            self.ext.close()


def append_ease_gpis():

    gpi_list = pd.read_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus.csv",index_col=0)

    gpi_list['ease_col'] = 0
    gpi_list['ease_row'] = 0

    LDAS = LDAS_io(exp='US_M36_SMOS40_noDA_cal_scaled')

    i = 0
    for idx, info in gpi_list.iterrows():
        i += 1
        print('%i / %i' % (i, len(gpi_list)))

        col, row = LDAS.grid.lonlat2colrow(gpi_list.loc[idx, 'lon'], gpi_list.loc[idx, 'lat'], domain=True)

        gpi_list.loc[idx,'ease_col'] = col
        gpi_list.loc[idx,'ease_row'] = row

    gpi_list.to_csv(r"D:\data_sets\ASCAT\warp5_grid\pointlist_warp_conus_w_ease_colrow.csv")


if __name__=='__main__':
    append_ease_gpis()























