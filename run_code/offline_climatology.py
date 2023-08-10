import xarray as xr
import numpy as np
from lib import driver
import pandas as pd
import os

#  initialize driver
LIMdriver = driver.Driver(f'namelist_cpc_offline_climatology.py')
# read in data
LIMdriver.get_variables(read=False)
vards = LIMdriver.use_vars['CPCtemp']['data']
# get coordinates and variables from varobject
lon = xr.DataArray(vards.lon,dims=('pts'))
lat = xr.DataArray(vards.lat,dims=('pts'))
climo = xr.DataArray(vards.climo,dims=('time', 'pts'))
time = pd.date_range(start=f'{vards.climoyears[0]}-01-01',end=f'{vards.climoyears[0]}-12-31')
# The exact year doesn't matter, the add_offset only uses doy to reference climo
# convert climatology to Kelvin
climo.values = climo.values + 273.15
climo.attrs['units'] = 'K'

if np.isnan(climo).any():
    print("The climo contains NaN values. Use 1D interpolation to fill the value")
    # I found out the three NaNs I have are (74N, 270E), (66N, 298E), and (22N, 276E). 
    # The first two are in Canadian Archipelago and last near cuba. 
    # Since these are out of the domain of interest, I simply interpolate the data using neighboring points. 

    for daily in climo:
        # Find indices of NaN values
        nan_indices = np.isnan(daily)
        # Create an array of non-NaN indices
        non_nan_indices = np.arange(len(daily))[~nan_indices]
        # Interpolate NaN values using adjacent points
        daily[nan_indices] = np.interp(np.where(nan_indices)[0], non_nan_indices, daily[non_nan_indices])



# save climatology to netcdf
dsclim = xr.Dataset({'T2m':climo,'time':time,'lat':lat, 'lon':lon})
fout = f'./data_clim/CPC.2p0.{vards.climoyears[0]}-{vards.climoyears[1]}.CYM.nc'
try: 
    os.remove(fout)
except OSError:
    pass

dsclim.to_netcdf(fout)