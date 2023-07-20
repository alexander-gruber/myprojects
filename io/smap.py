
import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict

from netCDF4 import Dataset, num2date, date2num

from myprojects.io.grids import EASE2
from myprojects.io.netcdf import ncfile_init

import h5py

class SMAP_io(object):

    def __init__(self):
        self.path = Path(r"D:\data_sets\SMAP\SPL2SMP.008\reformated")
        self.ds_ts = Dataset(r"D:\data_sets\SMAP\SPL2SMP.008\reformated\timeseries.nc")

        grid = EASE2('M36')
        self.lats = grid.ease_lats
        self.lons = grid.ease_lons

    def latlon2gpi(self, lat, lon):
        return np.argmin((self.lons - lon)**2 + (self.lats - lat)**2)

    def read(self, lat, lon, qc=True, rowcol=False):

        if rowcol:
            row, col = lat, lon
        else:
            row = np.argmin(np.abs(self.lats-lat))
            col = np.argmin(np.abs(self.lons-lon))

        dt = self.ds_ts['dt'][:, row, col]
        dt[dt.mask] = np.nan

        dates = self.ds_ts['time'][:]
        dates[~np.isnan(dt)] += dt[~np.isnan(dt)]
        smap_ts = pd.DataFrame(index=pd.DatetimeIndex(num2date(dates, self.ds_ts['time'].units,
                                                               only_use_python_datetimes=True, only_use_cftime_datetimes=False)))

        variables = ['soil_moisture', 'soil_moisture_error',
                     'retrieval_qual_flag', 'surface_flag',
                     'vegetation_opacity', 'vegetation_water_content',
                     'landcover_class', 'landcover_class_fraction']
        for var in variables:
            smap_ts[var] = self.ds_ts[var][:, row, col]

        smap_ts = smap_ts[~np.isnan(dt.data)]

        if qc:
            smap_ts = smap_ts[(smap_ts['surface_flag'] == 1024) | \
                              (smap_ts['retrieval_qual_flag'] == 0) | \
                              (smap_ts['retrieval_qual_flag'] == 8)]

        # if len(smap_ts) == 0:
            # print(f'No valid SMAP data for {lat:.2f} / {lon:.2f}')
            # smap_ts = None

        return smap_ts[~np.isnan(smap_ts['soil_moisture'])]

    def close(self):
        self.ds_ts.close()


def remove_corrupt_h5_files_l3():

    root = Path(r'D:\data_sets\SMAP\SPL3SMP.008\raw')
    paths = sorted(root.glob('*'))
    for path in paths:
        files = sorted(path.glob('*.h5'))
        if len(files) == 0:
            print(f'No files in {path.name}')
            path.unlink()
        elif len(files) > 1:
            for f in files[:-1]:
                trg = path.parents[1] / 'corrupt' / path.name
                if not trg.exists():
                    Path.mkdir(trg, parents=True)
                f.rename(path.parents[1] / 'corrupt' / path.name / f.name)
        else:
            continue

def create_L2_image_stack():

    root = Path(r"D:\data_sets\SMAP\SPL2SMP.008\raw")
    fout = root.parent / 'reformated' / 'images.nc'
    if not fout.parent.exists():
        Path.mkdir(fout.parent, parents=True)

    files = sorted(root.glob('**/*.h5'))
    t0s = pd.to_datetime([f.name[-29:-14] for f in files]).values
    u, i, c = np.unique(t0s, return_index=True, return_counts=True)
    t0s[i[c > 1]] += pd.Timedelta('1s') # sometimes Asc. and Dsc. files have same time stamp.
    dates = pd.Series(index=t0s).resample('12h').nearest().index

    grid = EASE2(gtype='M36')
    lats = grid.ease_lats
    lons = grid.ease_lons

    variables = ['dt',
                 'soil_moisture', 'soil_moisture_error',
                 'retrieval_qual_flag', 'surface_flag',
                 'vegetation_opacity', 'vegetation_water_content',
                 'landcover_class', 'landcover_class_fraction']

    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])
    with ncfile_init(fout, dimensions, variables) as ds:

        for i, (f, t0) in enumerate(zip(files,t0s)):
            print(f'{i} / {len(files)}')
            dts = t0 - dates
            t = np.argmin(np.abs(dts))
            with h5py.File(f, mode='r') as arr:
                rows = arr[f'Soil_Moisture_Retrieval_Data']['EASE_row_index'][:].astype('int')
                cols = arr[f'Soil_Moisture_Retrieval_Data']['EASE_column_index'][:].astype('int')
                for var in variables:
                    tmp_data = ds[var][t, :, :]
                    if var == 'dt':
                        tmp_data[rows, cols] = dts[t].total_seconds() / 3600 / 24
                        ds[var][t, :, :] = tmp_data
                    elif 'landcover' in var:
                        tmp_data[rows, cols] = arr[f'Soil_Moisture_Retrieval_Data'][var][:, 0]
                        ds[var][t, :, :] = tmp_data
                    else:
                        tmp_data[rows, cols] = arr[f'Soil_Moisture_Retrieval_Data'][var][:]
                        ds[var][t, :, :] = tmp_data


def create_L3_image_stack():

    root = Path(r"D:\data_sets\SMAP\SPL3SMP.008\raw")
    fout = root.parent / 'reformated' / 'images.nc'

    files = sorted(root.glob('**/*.h5'))
    dates = pd.to_datetime([f.parent.name for f in files])

    grid = EASE2(gtype='M36')
    lats = grid.ease_lats
    lons = grid.ease_lons

    variables = ['retrieval_qual_flag', 'surface_flag',
                 'vegetation_opacity', 'vegetation_water_content',
                 'landcover_class', 'landcover_class_fraction',
                 'soil_moisture', 'soil_moisture_error']

    file_vars = [f'{var}_{orbit}' for var in variables for orbit in ['am', 'pm']] + \
                ['soil_moisture','vegetation_opacity','vegetation_water_content','landcover_class', 'landcover_class_fraction']
    dimensions = OrderedDict([('time', dates), ('lat', lats), ('lon', lons)])
    with ncfile_init(fout, dimensions, file_vars) as ds:

        for t, f in enumerate(files):
            print(f'{t} / {len(files)}')

            with h5py.File(f, mode='r') as arr:
                for orbit in ['AM','PM']:
                    ext = '_pm' if orbit == 'PM' else ''
                    for var in variables:
                        if 'landcover' in var:
                            ds[f'{var}_{orbit.lower()}'][t, :, :] = arr[f'Soil_Moisture_Retrieval_Data_{orbit}'][f'{var}{ext}'][:, :, 0]
                        else:
                            ds[f'{var}_{orbit.lower()}'][t, :, :] = arr[f'Soil_Moisture_Retrieval_Data_{orbit}'][f'{var}{ext}'][:, :]

                    mask_am = (arr[f'Soil_Moisture_Retrieval_Data_AM'][f'surface_flag'][:, :] != 1024) & \
                              (arr[f'Soil_Moisture_Retrieval_Data_AM'][f'retrieval_qual_flag'][:, :] != 0) & \
                              (arr[f'Soil_Moisture_Retrieval_Data_AM'][f'retrieval_qual_flag'][:, :] != 8)
                    mask_pm = (arr[f'Soil_Moisture_Retrieval_Data_PM'][f'surface_flag_pm'][:, :] != 1024) & \
                              (arr[f'Soil_Moisture_Retrieval_Data_PM'][f'retrieval_qual_flag_pm'][:, :] != 0) & \
                              (arr[f'Soil_Moisture_Retrieval_Data_PM'][f'retrieval_qual_flag_pm'][:, :] != 8)

                    tmp_am = arr[f'Soil_Moisture_Retrieval_Data_AM'][f'soil_moisture'][:, :]
                    tmp_pm = arr[f'Soil_Moisture_Retrieval_Data_PM'][f'soil_moisture_pm'][:, :]
                    tmp_am[mask_am] = np.nan
                    tmp_pm[mask_pm] = np.nan
                    tmp_am[tmp_am == -9999.] = np.nan
                    tmp_pm[tmp_pm == -9999.] = np.nan
                    ds['soil_moisture'][t, :, :] = np.nanmean([tmp_am, tmp_pm],axis=0)

                    tmp_am = arr[f'Soil_Moisture_Retrieval_Data_AM'][f'vegetation_opacity'][:, :]
                    tmp_pm = arr[f'Soil_Moisture_Retrieval_Data_PM'][f'vegetation_opacity_pm'][:, :]
                    tmp_am[mask_am] = np.nan
                    tmp_pm[mask_pm] = np.nan
                    tmp_am[tmp_am == -9999.] = np.nan
                    tmp_pm[tmp_pm == -9999.] = np.nan
                    ds['vegetation_opacity'][t, :, :] = np.nanmean([tmp_am, tmp_pm],axis=0)

                    tmp_am = arr[f'Soil_Moisture_Retrieval_Data_AM'][f'vegetation_water_content'][:, :]
                    tmp_pm = arr[f'Soil_Moisture_Retrieval_Data_PM'][f'vegetation_water_content_pm'][:, :]
                    tmp_am[mask_am] = np.nan
                    tmp_pm[mask_pm] = np.nan
                    tmp_am[tmp_am == -9999.] = np.nan
                    tmp_pm[tmp_pm == -9999.] = np.nan
                    ds['vegetation_water_content'][t, :, :] = np.nanmean([tmp_am, tmp_pm],axis=0)

                    tmp_am = arr[f'Soil_Moisture_Retrieval_Data_AM'][f'landcover_class'][:, :, 0]
                    tmp_pm = arr[f'Soil_Moisture_Retrieval_Data_PM'][f'landcover_class_pm'][:, :, 0]
                    tmp_am[mask_am] = 254
                    tmp_pm[mask_pm] = 254
                    tmp_data = np.full(tmp_am.shape, 254)
                    tmp_data[(tmp_am != 254) & (tmp_pm == 254)] = tmp_am[(tmp_am != 254) & (tmp_pm == 254)]
                    tmp_data[(tmp_am == 254) & (tmp_pm != 254)] = tmp_pm[(tmp_am == 254) & (tmp_pm != 254)]
                    tmp_data[(tmp_am != 254) & (tmp_pm != 254)] = tmp_am[(tmp_am != 254) & (tmp_pm != 254)]
                    ds['landcover_class'][t, :, :] = tmp_data

                    tmp_am = arr[f'Soil_Moisture_Retrieval_Data_AM'][f'landcover_class_fraction'][:, :, 0]
                    tmp_pm = arr[f'Soil_Moisture_Retrieval_Data_PM'][f'landcover_class_fraction_pm'][:, :, 0]
                    tmp_am[mask_am] = 254
                    tmp_pm[mask_pm] = 254
                    tmp_data = np.full(tmp_am.shape, 254)
                    tmp_data[(tmp_am != 254) & (tmp_pm == 254)] = tmp_am[(tmp_am != 254) & (tmp_pm == 254)]
                    tmp_data[(tmp_am == 254) & (tmp_pm != 254)] = tmp_pm[(tmp_am == 254) & (tmp_pm != 254)]
                    tmp_data[(tmp_am != 254) & (tmp_pm != 254)] = tmp_am[(tmp_am != 254) & (tmp_pm != 254)]
                    ds['landcover_class_fraction'][t, :, :] = tmp_data


if __name__=='__main__':
    # create_L2_image_stack()
    # create_L3_image_stack()
    # remove_corrupt_h5_files_l2()

    pass

# ncks -4 -L 4 --cnk_dmn time,10000 --cnk_dmn lat,1 --cnk_dmn lon,1 images.nc timeseries.nc

