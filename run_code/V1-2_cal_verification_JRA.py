#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import sys
import lib
from lib.tools import check_lat_order


# In[2]:

dirVeri = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification'

# varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
varnames = ["H500"]

for varname in varnames:

    varname = varname
    fileVarname = 'anomaly'
    full_years = list(range(1958, 2024))
    # full_years = list(range(1958, 1959))

    mask = xr.open_dataset(f'{dirVeri}/mask.nc')
    mask = check_lat_order(mask)
    mask = mask['mask']


    for year in full_years:
        print(f'----{year}----')
        # We need current year + the year after because some of the week34 verification will be in the next year
        ds_list = [f'{dirVeri}/{varname}/links/{varname}.{i}.nc' for i in range(year,year+2)]
        print(ds_list)
        ds = xr.open_mfdataset(ds_list, combine='nested', concat_dim='time')
        ds = ds[fileVarname]
        ds = check_lat_order(ds)

        # dates_in_the_year = ds['time'].sel(time=slice(f'{year}-01-01',f'{year}-12-31'))
        start_date = f'{year}-01-01'
        end_date   = f'{year}-12-31'
        dates_in_the_year = pd.date_range(start=start_date, end=end_date)

        list_week34_data = []

        for i, date in enumerate(dates_in_the_year):
            week3 = date + timedelta(days=21)
            week4 = date + timedelta(days=28)
            week34 = [str(week3), str(week4)]
            # if i%100 == 0: 
            #     print(f'processing {date}')
            #     print(f'week34 = {week34}')
            
            # Data in the T2m.year.nc files are already 7-day running mean, so we just need to average week3 and 4.
            ds_week34_tmp = ds.sel(time=week34).mean(dim='time')
            list_week34_data.append(ds_week34_tmp)

        ds_week34= xr.concat(list_week34_data, dim='time')
        ds_week34 = ds_week34.sortby('time') # time is 0-364 here
        ds_week34_mask = ds_week34.where(~mask.isnull())

        ds_week34 = xr.Dataset(
            {varname:ds_week34_mask,f'{varname}_full':ds_week34},
            coords={'time':dates_in_the_year,'lat':ds_week34.lat,'lon':ds_week34.lon
            }
        )
        # print(ds_week34)
        fout = f'{dirVeri}/{varname}/{varname}.{year}.week34.nc'
        os.system(f'rm -f {fout}')
        ds_week34.to_netcdf(fout)


 




