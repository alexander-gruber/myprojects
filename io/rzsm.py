# -*- coding: utf-8 -*-
# Copyright (c) 2021, Vienna University of Technology (TU Wien),
# Department of Geodesy and Geoinformation (GEO).
# All rights reserved.
#
# All information contained herein is, and remains the property of Vienna
# University of Technology (TU Wien), Department of Geodesy and Geoinformation
# (GEO). The intellectual and technical concepts contained herein are
# proprietary to Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO). Dissemination of this information or
# reproduction of this material is forbidden unless prior written permission
# is obtained from Vienna University of Technology (TU Wien), Department of
# Geodesy and Geoinformation (GEO).

"""
Module for reading GLDAS data
"""

from pathlib import Path

import pynetcf.time_series as nc_dataset

from smecv_grid.grid import SMECV_Grid_v052


def get_smecv_grid(version='v052'):
    """
    Allows for different versions of the grid to be chosen. To be updated as and when smecv-grid package is changed.
    :param version: string: string representing the version number
    :return: Function from the smecv-grid.grid file
    """
    grids = {'v052': SMECV_Grid_v052}

    return grids.get(version, SMECV_Grid_v052)


class RZSM_io(nc_dataset.GriddedNcTs):

    def __init__(self, path=None, grid=None, parameter='rzsm_1', read_bulk=True, fn_format='{:04d}'):
        """
        Reads GLDAS time series for GLDAS v2.1 for use in the CCI
        :param path: str: path to the data root directory
        :param grid:
        :param parameter: string or list, optional
        one or list of ['057', '065', '085_L1', '086_L1', '086_L2',
                        '086_L3', '086_L4', '131', '132', '138']
        parameters to read, see wiki for more information
        Default: '086_L1' soil moisture layer 1
        :param read_bulk: boolean, optional
        if True the whole cell is read at once which makes bulk processing
        tasks that need all the time series of a cell faster.
        Default: False
        :param mask_frozen:
        :param celsius: boolean, optional: if True temperature values are returned in degrees Celsius, otherwise Kelvin
        :param fn_format:
        """

        # Set up grid
        if grid is None:
            grid = get_smecv_grid('v052')()

        if type(parameter) != list:
            parameter = [parameter]
        self.parameters = parameter

        offsets = dict()
        self.path = dict()

        if not path:
            self.path['ts'] = Path(r"M:\Projects\CCIplus_Soil_Moisture\07_data\ESA_CCI_SM_v09.1\044_rzsm_ts")
        else:
            self.path['ts'] = path


        super(RZSM_io, self).__init__(self.path['ts'], grid=grid, ioclass=nc_dataset.ContiguousRaggedTs,
                                          ioclass_kws={'read_bulk': read_bulk}, parameters=self.parameters,
                                          offsets=offsets, fn_format=fn_format)

    def read(self, *args, **kwargs):


        data = super(RZSM_io, self).read(*args, **kwargs)

        # data['gldas'] = data['SoilMoi0_10cm_inst']
        # del data['SoilMoi0_10cm_inst']
        data = data.tz_localize(None)

        return data.dropna()

if __name__=='__main__':

    io = RZSM_io(parameter=['rzsm_1','rzsm_2','rzsm_3','rzsm_4'])

    lat, lon = 52.20328454384528, 6.310342759561394
    data = io.read(lon, lat)
    io.grid.find_nearest_gpi(lat, lon )
    print(data)