
import pandas as pd

from smecv.input.lprm import Lprm

class AMSR2_io(object):

    def __init__(self):
        self.io_asc = Lprm(
            r'D:\data_sets\AMSR2\AMSR2_S3_VEGC_v070_until202112\time_series\a',
            sensor='amsr2')
        self.io_dsc = Lprm(
            r'D:\data_sets\AMSR2\AMSR2_S3_VEGC_v070_until202112\time_series\d',
            sensor='amsr2')
        self.grid = self.io_asc.grid

    def read(self, *args, dsc_only=False):

        if len(args) == 1:
            gpi = int(args[0])
        else:
            gpi = self.grid.find_nearest_gpi(lon, lat)[0]

        try:
            ds_dsc = self.io_dsc._read_gp(gpi)
            ds_dsc = ds_dsc[(ds_dsc['flag'] == 0) | (ds_dsc['flag'] == 64)]['sm']
            if dsc_only:
                df = ds_dsc
            else:
                ds_asc = self.io_asc._read_gp(gpi)
                ds_asc = ds_asc[(ds_asc['flag'] == 0) | (ds_asc['flag'] == 64)]['sm']
                df = pd.concat((ds_asc, ds_dsc)).sort_index()
        except:
            df = pd.Series(dtype='float')

        return df