
import pyproj
import logging

import numpy as np

class EASE2(object):
    """
    Class that contains EASE2 grid parameters

    Parameters
    ----------
    gtype : string
        Grid type (M03, M09 or M36). If not provided, grid type will be derived from tilegrids

    Attributes
    ----------
    ease_lats : np.array
        Array containing the latitudes of grid rows
    ease_lons : np.array
        Array containing the longitudes of grid columns
    shape : tuple
        lat/lon dimensions
    """

    def __init__(self, gtype):

       self.setup_grid(gtype)


    def setup_grid(self, gridtype):

        if gridtype == 'M36':
            map_scale = 36032.220840584
        elif gridtype == 'M25':
            map_scale = 25025.26 # fewer digits are defined by NSIDC to avoid projection issues at +-180 degree
        elif gridtype == 'M09':
            map_scale = 9008.055210146
        elif gridtype == 'M03':
            map_scale = 3002.6850700487
        else:
            raise NotImplementedError(
                "Only M03, M09 and M36 grids supported .")

        # Upper boundary of north(south)-most M36 pixel. Used for cutting all valid M03/M09 pixels above (below).
        latmax = 85.04457

        ease = pyproj.Proj(("+proj=cea +lat_0=0 +lon_0=0 +lat_ts=30 "
                                 "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m"))

        x_min, y_max = ease(-180, 90)
        x_max, y_min = ease(180, -90)

        # Calculate number of grid cells in x-dimension. Map scale is defined such that an exact integer number of
        # grid cells fits in longitude direction.
        x_extent = x_max - x_min
        x_dim = round(x_extent / map_scale)

        # Calculate exact x and y dimensions accounting for rounding error in map-scale (in the 10th digit or so)
        x_pix_size = x_extent / x_dim
        y_pix_size = (map_scale ** 2) / x_pix_size

        # Generate arrays with all x/y center coordinates in map-space, centered around 0
        x_arr_pos = np.arange(x_pix_size / 2, x_max, x_pix_size)
        x_arr_neg = np.arange(-x_pix_size / 2, x_min, -x_pix_size)
        x_arr = np.concatenate([x_arr_neg[::-1], x_arr_pos])

        y_arr_pos = np.arange(y_pix_size / 2, y_max, y_pix_size)
        y_arr_neg = np.arange(-y_pix_size / 2, y_min, -y_pix_size)
        y_arr = np.concatenate([y_arr_pos[::-1], y_arr_neg])

        # Clip all valid grid cells that are above (below) the last valid M36 grid cell
        i_valid = np.where(ease(np.zeros(y_arr.shape), y_arr + y_pix_size/2, inverse=True)[1] < latmax)[0][0]
        y_arr = y_arr[i_valid:-i_valid]

        # Convert all grid cell coordinates from map-space to lat/lon
        self.ease_lons = ease(x_arr, np.zeros(x_arr.shape), inverse=True)[0]
        self.ease_lats = ease(np.zeros(y_arr.shape), y_arr, inverse=True)[1]


    def colrow2lonlat(self, col, row):
        """ Convert col/row (domain-based) into lon/lat """
        return self.ease_lons[col], self.ease_lats[row]

    def lonlat2colrow(self, lon, lat):
        """ Find nearest GLOBAL tile (col/row) from any given lon/lat """
        londif = np.abs(self.ease_lons - lon)
        latdif = np.abs(self.ease_lats - lat)
        col = np.where(np.abs(londif-londif.min())<0.0001)[0][0]
        row = np.where(np.abs(latdif-latdif.min())<0.0001)[0][0]

        return col, row
