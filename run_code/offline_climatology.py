import xarray as xr
import numpy as np
from lib import driver
import pandas as pd
import os

resolution = 2
#  initialize driver
LIMdriver = driver.Driver(f'namelist_cpc_offline_climatology.py')
# read in data
LIMdriver.get_variables(read=False)
# LIMdriver.get_variables(read=True)

# varname = 'CPCtemp'
for varname in LIMdriver.use_vars.keys():
    vards = LIMdriver.use_vars[varname]['data']
    # get coordinates and variables from varobject
    lon = xr.DataArray(vards.lon,dims=('pts'))
    lat = xr.DataArray(vards.lat,dims=('pts'))

    climo = xr.DataArray(vards.climo,dims=('time', 'pts'))
    time = pd.date_range(start=f'{vards.climoyears[0]}-01-01',end=f'{vards.climoyears[0]}-12-31')
    # The exact year doesn't matter, the add_offset only uses doy to reference climo
    # convert climatology to Kelvin



    if varname == 'CPCtemp':
        climo.values = climo.values + 273.15
        climo.attrs['units'] = 'K'
    if varname == 'T2m':
        climo.attrs['units'] = 'K'

    # if np.isnan(climo).any():
    #     print("The climo contains NaN values. Use 1D interpolation to fill the value")
    #     # I found out the three NaNs I have are (74N, 270E), (66N, 298E), and (22N, 276E). 
    #     # The first two are in Canadian Archipelago and last near cuba. 
    #     # Since these are out of the domain of interest, I simply interpolate the data using neighboring points. 

    #     for daily in climo:
    #         # Find indices of NaN values
    #         nan_indices = np.isnan(daily)
    #         # Create an array of non-NaN indices
    #         non_nan_indices = np.arange(len(daily))[~nan_indices]
    #         # Interpolate NaN values using adjacent points
    #         daily[nan_indices] = np.interp(np.where(nan_indices)[0], non_nan_indices, daily[non_nan_indices])


    def cyclic_running_mean(data, window_size):
    
        # Create a cyclic extension of the data to handle boundary values
        cyclic_data = np.concatenate((data[-window_size:,:], data[:,:]),axis=0)
        print(cyclic_data.shape)
        # Initialize an array to store the running mean
        running_mean = np.zeros(data.shape)
        print(data.shape)    
        # Calculate the running mean
        print(f'calculate {window_size}-day running mean')
        for i in range(data.shape[0]):
            # print(i,i+window_size)
            running_mean[i,:] = np.mean(cyclic_data[i:i+window_size,:],axis=0)# python takes i - i+windown_size-1
        
        return running_mean

    climo_running_mean = cyclic_running_mean(climo,LIMdriver.time_window)
    climo = xr.DataArray(climo_running_mean,dims=('time', 'pts'))

    print(climo.shape)
    # save climatology to netcdf
    dsclim = xr.Dataset({varname:climo,'time':time,'lat':lat, 'lon':lon})
    # fout = f'./data_clim/CPC.{resolution}p0.{vards.climoyears[0]}-{vards.climoyears[1]}.CYM.nc'
    fout = f'{LIMdriver.VAR_FILE_PREFIX}{varname}.nc'
    try: 
        os.remove(fout)
    except OSError:
        pass

    dsclim.to_netcdf(fout)