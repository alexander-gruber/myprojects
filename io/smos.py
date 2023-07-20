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


def main():
    data_path = Path(r"D:\data_sets\SMOS\timeseries_processed21102022\SMOSL2_v700")
    grid = load_grid(data_path / "grid.nc")

    reader = GriddedNcOrthoMultiTs(data_path, grid=grid)

    # all gridpoints in cell 1359 (central europe, see https://github.com/TUW-GEO/smecv-grid/blob/master/docs/5x5_cell_partitioning_cci.png)
    gpis, lat, lon = grid.grid_points_for_cell(1359)

    creader = ColumnCombineAdapter(reader,
                                   comb_rfi,
                                   func_kwargs={'axis': 1},
                                   columns=['N_RFI_X', 'N_RFI_Y', 'M_AVA0'],
                                   new_name="COMBINED_RFI")

    # adapt the timestamps ----------------------
    tadapt_kwargs = {
        'time_offset_fields': 'Seconds',
        'time_units': 's',
        'base_time_field': 'Days',
        'base_time_units': 'ns',
        'base_time_reference': '2000-01-01',
    }

    treader = TimestampAdapter(creader, **tadapt_kwargs)

    # Read example
    # data = treader.read(gpis[117])
    data = treader.read(gpis[117])

    print(data['Soil_Moisture'])

if __name__=='__main__':
    main()