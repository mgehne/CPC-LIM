import xarray as xr
import numpy as np
from lib import driver
import pandas as pd


#  initialize driver
LIMdriver = driver.Driver(f'namelist_clim.py')
# read in data
LIMdriver.get_variables(read=False)
vards = LIMdriver.use_vars['CPCtemp']['data']
# get coordinates and variables from varobject
lon = xr.DataArray(vards.lon,dims=('pts'))
lat = xr.DataArray(vards.lat,dims=('pts'))
climo = xr.DataArray(vards.climo,dims=('time', 'pts'))
time = pd.date_range(start='1981-01-01',end='1981-12-31')
# convert climatology to Kelvin
climo.values = climo.values + 273.15
climo.attrs['units'] = 'K'
# save climatology to netcdf
dsclim = xr.Dataset({'T2m':climo,'time':time,'lat':lat, 'lon':lon})
dsclim.to_netcdf('./data_clim/CPC.1981-2010.nc')