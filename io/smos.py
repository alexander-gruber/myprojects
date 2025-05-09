from pathlib import Path

from pynetcf.time_series import GriddedNcOrthoMultiTs
from pygeogrids.netcdf import load_grid

from pytesmo.validation_framework.adapters import ColumnCombineAdapter, TimestampAdapter

# get the daily RFI from (N_RFI_X + N_RFI_Y) / M_AVA0 ----------------------
def comb_rfi(row):
    # 'COMBINED_RFI' is created using the formula:
    # (N_RFI_X + N_RFI_Y) / M_AVA0 and the class
    # pytesmo.validation_framework.adapters.ColumnCombineAdapter
    return (row['N_RFI_X'] + row['N_RFI_Y']) / row['M_AVA0']


from pathlib import Path

import matplotlib.pyplot as plt
import pygeogrids.netcdf as ncgrid
import pynetcf.time_series as nc_dataset


class SMOS_io(nc_dataset.GriddedNcTs):

    def __init__(self, path=None, grid=None, parameter='Soil_Moisture', read_bulk=True, fn_format='{:04d}'):
        """
        Reads ERA5-Land  time series
        """

        if grid is None:
            grid = ncgrid.load_grid(r"M:\Projects\QA4SM_HR\07_data\SERVICE_DATA\SMOS_L2\SMOSL2_v700\grid.nc")

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter
        self.path = dict()

        if not path:
            self.path['ts'] = Path(r"M:\Projects\QA4SM_HR\07_data\SERVICE_DATA\SMOS_L2\SMOSL2_v700")
        else:
            self.path['ts'] = path

        # !!! DOESN'T WORK YET, NOT SURE WHICH IOCLASS SHOULD BE USED...
        super(SMOS_io, self).__init__(self.path['ts'], grid=grid, ioclass=nc_dataset.GriddedNcIndexedRaggedTs,
                                          ioclass_kws={'grid': grid}, parameters=self.parameters,
                                          fn_format=fn_format)


    def read(self, *args, **kwargs):
        if 'only_valid' in kwargs:
            del kwargs['only_valid']

        res = super(SMOS_io, self).read(*args, **kwargs)
        # ['tp'].resample('1D').sum() * 1000 # [m] to [mm]
        res.name = 'precipitation'
        return res



# def main():
    # data_path = Path(r"D:\data_sets\SMOS\timeseries_processed21102022\SMOSL2_v700")
    # grid = load_grid(data_path / "grid.nc")
    #
    # reader = GriddedNcOrthoMultiTs(data_path, grid=grid)
    #
    # # all gridpoints in cell 1359 (central europe, see https://github.com/TUW-GEO/smecv-grid/blob/master/docs/5x5_cell_partitioning_cci.png)
    # gpis, lat, lon = grid.grid_points_for_cell(1359)
    #
    # creader = ColumnCombineAdapter(reader,
    #                                comb_rfi,
    #                                func_kwargs={'axis': 1},
    #                                columns=['N_RFI_X', 'N_RFI_Y', 'M_AVA0'],
    #                                new_name="COMBINED_RFI")
    #
    # # adapt the timestamps ----------------------
    # tadapt_kwargs = {
    #     'time_offset_fields': 'Seconds',
    #     'time_units': 's',
    #     'base_time_field': 'Days',
    #     'base_time_units': 'ns',
    #     'base_time_reference': '2000-01-01',
    # }
    #
    # treader = TimestampAdapter(creader, **tadapt_kwargs)
    #
    # # Read example
    # # data = treader.read(gpis[117])
    # data = treader.read(gpis[117])
    #
    # print(data['Soil_Moisture'])

if __name__=='__main__':

    lat, lon = 31.5, -83.55
    smos = SMOS_io()

    ts = smos.read(lon, lat)
    print(ts)