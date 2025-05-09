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
Module for reading ERA5 data
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pygeogrids.netcdf as ncgrid
import pynetcf.time_series as nc_dataset


class E5L_io(nc_dataset.GriddedNcTs):

    def __init__(self, path=None, grid=None, parameter='tp', read_bulk=True, fn_format='{:04d}'):
        """
        Reads ERA5-Land  time series
        """

        if grid is None:
            grid = ncgrid.load_grid(r"M:\Datapool\ECMWF_reanalysis\02_processed\ERA5-Land\datasets\sm_precip_lai\grid.nc")

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter
        self.path = dict()

        if not path:
            self.path['ts'] = Path(r"M:\Datapool\ECMWF_reanalysis\02_processed\ERA5-Land\datasets\sm_precip_lai")
        else:
            self.path['ts'] = path

        super(E5L_io, self).__init__(self.path['ts'], grid=grid, ioclass=nc_dataset.OrthoMultiTs,
                                          ioclass_kws={'read_bulk': read_bulk}, parameters=self.parameters,
                                          fn_format=fn_format)


    def read(self, *args, **kwargs):
        if 'only_valid' in kwargs:
            del kwargs['only_valid']

        res = super(E5L_io, self).read(*args, **kwargs)['tp'].resample('1D').sum() * 1000 # [m] to [mm]
        res.name = 'precipitation'
        return res

    def read_sm(self, *args, **kwargs):
        if 'only_valid' in kwargs:
            del kwargs['only_valid']

        res = super(E5L_io, self).read(*args, **kwargs)['swvl2'].resample('1D').sum()
        res.name = 'soil_moisture'
        return res
