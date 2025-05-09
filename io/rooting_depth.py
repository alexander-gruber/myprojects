
import numpy as np
import rasterio

class RD_io(object):

    def __init__(self):

        file_path = r"H:\work\experiments\palmer_model\Effective_Rooting_Depth.tif"
        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)
            profile = dataset.profile
            image[image==profile['nodata']] = np.nan

        self.rd = image
        self.lats = np.arange(89.75,-90,-0.5)
        self.lons = np.arange(-179.75,180,0.5)

    def read(self, lon, lat):
        row = np.argmin((self.lats-lat)**2)
        col = np.argmin((self.lons-lon)**2)
        res = self.rd[row, col] * 1000.
        if np.isnan(res):
            res = 500.
        return res


