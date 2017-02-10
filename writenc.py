import numpy as np
from netCDF4 import Dataset

def writenc(filename,lat,lon,tstamp,val):

 ni=lon.shape[0]
 nj=lat.shape[0]

 rootgrp = Dataset(filename, 'w', format='NETCDF3_64BIT')
 lats = rootgrp.createDimension('LAT', nj)
 lons = rootgrp.createDimension('LON', ni)
 time = rootgrp.createDimension('TIME', None)


 longitudes = rootgrp.createVariable('LON','f8',('LON',))
 latitudes = rootgrp.createVariable('LAT','f8',('LAT',))
 times = rootgrp.createVariable('TIME','f8',('TIME',))
 levels = rootgrp.createVariable('HA','f8',('TIME','LAT','LON'))

 rootgrp.description = ''
 rootgrp.history = 'JRC Ispra European Commission'
 rootgrp.source = 'netCDF4 python module tutorial'
 latitudes.units = 'degrees_north'
 latitudes.point_spacing = 'even'
 longitudes.units = 'degrees_east'
 longitudes.point_spacing = 'even'
 levels.units = 'm'
 times.units = 'seconds'


 levels[:]=val
 times[:]=tstamp
 latitudes[:]=lat
 longitudes[:]=lon

 rootgrp.close()



