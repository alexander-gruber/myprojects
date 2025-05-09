import os
import warnings
warnings.filterwarnings("ignore")

from myprojects.io.ascat import HSAF_io
from pynetcf.time_series import GriddedNcOrthoMultiTs, GriddedNcIndexedRaggedTs, GriddedNcContiguousRaggedTs
from pygeogrids.netcdf import load_grid

class reader(object):

    def __init__(self, name, path=None, ioclass=None):

        if name.upper() == 'GLDAS':
            tmppath = r"D:\data_sets\GLDAS\GLDAS_NOAH_025_3H.2.1\netcdf"
            tmpioclass = GriddedNcOrthoMultiTs
        elif name.upper() == 'SMAP':
            tmppath = r"D:\data_sets\SMAP\SMAPL2_V8"
            tmpioclass = GriddedNcOrthoMultiTs
        elif name.upper() == 'SMOS':
            tmppath = r"D:\data_sets\SMOS\SMOSL2_v700"
            tmpioclass = GriddedNcIndexedRaggedTs
        elif name.upper() == 'AMSR2':
            tmppath = r"D:\data_sets\AMSR2\AMSR2_S3_VEGC_LPRMv7.2"
            tmpioclass = GriddedNcContiguousRaggedTs
        elif name.upper() == 'ASCAT':
            tmppath = r"D:\data_sets\HSAF"
            tmpioclass = None
        else:
            print('Unknown data set!')
            return

        if not ioclass:
            ioclass = tmpioclass
        if path:
            self.path = path
        else:
            self.path = tmppath

        if name.upper() == 'ASCAT':
            self.obj = HSAF_io()
            self.grid = self.obj.grid
        else:
            self.grid = load_grid(os.path.join(self.path,'grid.nc'))
            self.obj = ioclass(self.path, self.grid, ioclass_kws={'read_bulk': True})

        self.name = name.upper()

    def read(self, *args, mask=True, resample=True, **kwargs):

        try:
            data = self.obj.read(*args, **kwargs)

            if mask:
                if self.name == 'GLDAS':
                    data = data[(data['SWE_inst'] < 0.001) & (data['SoilTMP0_10cm_inst'] > 1.)]['SoilMoi0_10cm_inst']
                elif self.name == 'SMAP':
                    data = data[(data['quality_flag']==0)|(data['quality_flag']==8)]['soil_moisture']
                elif self.name == 'SMOS':
                    data = data[(data['RFI_Prob'] < 0.2) & (data['Chi_2_P'] > 0.05)]['Soil_Moisture']
                elif self.name == 'AMSR2':
                    data = data[data['flag']==0]['sm']

            if resample:
                # if self.name == 'GLDAS':
                #     data = data[data.index.hour == 0]
                data = data.resample('1D').mean()

            data.name = self.name
            return data.dropna()

        except:
            return None