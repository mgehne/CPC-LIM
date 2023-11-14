import xarray as xr
import os
from os import system
import glob
import numpy as np
from datetime import datetime as dt,timedelta

sliding_climo=True
# sliding_climo=False

# subsetting_data = False
subsetting_data = True

# expt_name = '10a_using_Sam_rawdata'
# expt_name = '9d_sliding_climo_5_deg'
# expt_name = '10d_sliding_climo'
expt_name = 'hindcast_fold_10'
if sliding_climo:
    # in_data_path  = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}' # for sliding climo
    in_data_path  = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b2_sliding_climo_no_double_running_mean' # for sliding climo??
else:
    in_data_path  = '/data/ycheng/JRA/Data'
# out_data_path = '/scratch/ycheng/JRA/Data/data_retrospective/test'
# out_data_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective_subset'
out_data_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective'
os.system(f'mkdir -p {out_data_path}')

if sliding_climo:
    RT_VARS = ['T2m','SOIL','SLP','colIrr','H500','SST','SF100','SF750']
else:
    RT_VARS = ['surf','phy2m','land','sst','sf','hgt']

# RT_VARS = ['T2m'] 
years   = dt
for var in RT_VARS:
    # os.system(f'rm -r {out_data_path}/{var}')
    # os.system(f'mkdir {out_data_path}/{var}')
    # for year in range(2016,2024,1):
    # for year in range(2023,2024,1):
    for year in range(2011,2012,1):
    # for year in range(1995,2015,1):
        if sliding_climo:
            os.system(f'ln -s {in_data_path}/{year}/{var}/{var}.{year}.nc {out_data_path}/{var}')
            # os.system(f'ln -s {in_data_path}/{year}_no_mask/{var}/{var}.{year}.nc {out_data_path}/{var}')
            # print(f'ln -s {in_data_path}/{year}_no_mask/{var}/{var}.{year}.nc')
        else:
            if var=='hgt' or var=='sf':
                os.system(f'ln -s {in_data_path}/{year}/{var}_{year}_1p25.nc {out_data_path}/{var}')
            else:
                os.system(f'ln -s {in_data_path}/{year}/{var}_{year}.nc {out_data_path}/{var}')

    print(f'--------- processing {var} ---------')
    filenames = sorted(glob.glob(f'{out_data_path}/{var}/{var}*nc'))
    ds = xr.open_mfdataset(filenames, combine='nested', concat_dim='time')
    
    if subsetting_data:
        T_START = dt(2011,1,1) #dt(YEAR,MONTH,1) 
        # T_END = dt(2017,1,20) #dt(YEAR,MONTH,LASTDAY)
        T_END = dt(2011,1,31) #dt(YEAR,MONTH,LASTDAY)
        selected_dates = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
        ds = ds.sel(time=selected_dates)
    
    print(ds['time'][0:10],ds['time'][-10::])
    # try:
        # os.system(f'rm {out_data_path}/{var}All.nc')
        # os.system(f'rm {out_data_path}/{var}All.nc')
    # except OSError:
        # pass
    ds.to_netcdf(f'{out_data_path}/{var}All_subset.nc')
    # # ds.to_netcdf(f'{out_data_path}/{var}All.nc')
    