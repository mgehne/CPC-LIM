import numpy as np
import xarray as xr
import time
import os
import warnings
warnings.filterwarnings('ignore') 
# This is to suppress the warning of RuntimeWarning: Mean of empty slice due to NAN

years = np.arange(1979,2025,1)
# years = np.arange(2020,2025,1)
diri='/Users/ycheng/CPC/Data/cpcdata/Global'
resolution = 2
for year in years:
    # Record the start time
    start_time = time.time()
    ds1 = xr.open_mfdataset(f'{diri}/tmin.{year}.nc')
    ds2 = xr.open_mfdataset(f'{diri}/tmax.{year}.nc')# cpc files are in degC
    ds3 = ds1.merge(ds2,compat='no_conflicts')
    ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
    obs = ds4.drop_vars(('tmin','tmax'))

    # CPC data is -89.75 to 89.75 and the new grid is -90 to 90
    # Interpolating here because the interpolation in dataset.py does not allow extrapolation.
      
    new_lats = np.arange(-90,91,resolution)
    new_lons = np.arange(0,360,resolution)
    obs = obs.interp(lat=new_lats,lon=new_lons)
    
    fout = f'{diri}/tavg.{year}.{resolution}p0.nc'
    os.system('rm -f {fout}')
    obs.to_netcdf(fout)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"For {year}, elapsed time: {elapsed_time} seconds")