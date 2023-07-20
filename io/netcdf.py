
from netCDF4 import Dataset, date2num

def ncfile_init(fname, dimensions, variables):

    ds = Dataset(fname, mode='w')
    timeunit = 'days since 1900-01-01 00:00'

    # Initialize dimensions
    chunksizes = []
    for dim in dimensions:

        # convert pandas Datetime Index to netCDF-understandable numeric format
        if dim == 'time':
            dimensions[dim] = date2num(dimensions[dim].to_pydatetime(), timeunit)

        # Files are per default image chunked
        if dim in ['lon', 'lat']:
            chunksize = len(dimensions[dim])
        else:
            chunksize = 1
        chunksizes.append(chunksize)

        dtype = dimensions[dim].dtype
        ds.createDimension(dim, len(dimensions[dim]))
        ds.createVariable(dim, dtype,
                          dimensions=(dim,),
                          chunksizes=(chunksize,),
                          zlib=True)
        ds.variables[dim][:] = dimensions[dim]

    # Coordinate attributes following CF-conventions
    if 'time' in dimensions:
        ds.variables['time'].setncatts({'long_name': 'time',
                                        'units': timeunit})
    ds.variables['lon'].setncatts({'long_name': 'longitude',
                                   'units': 'degrees_east'})
    ds.variables['lat'].setncatts({'long_name': 'latitude',
                                   'units': 'degrees_north'})

    # Initialize variables
    for var in variables:
        ds.createVariable(var, 'float32',
                          dimensions=list(dimensions.keys()),
                          chunksizes=chunksizes,
                          fill_value=-9999.,
                          zlib=True)

    return ds

