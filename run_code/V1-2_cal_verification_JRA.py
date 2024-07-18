#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import sys
import lib
from lib.tools import check_lat_order
from datetime import datetime as dt,timedelta


# In[2]:

dirVeri = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification'

# varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
varnames = ["T2m"]
# varnames = ["H500"]
# add_offset = False
add_offset = True


for varname in varnames:

    varname = varname
    fileVarname = 'anomaly'
    full_years = list(range(2017, 2018))
    # full_years = list(range(2017, 2023))
    # full_years = list(range(1958, 2023))
    # full_years = list(range(1958, 1979))

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

        list_week3_data = []
        list_week4_data = []
        list_week34_data = []

        if add_offset:
            ds_oldclim = xr.open_dataset(f'{dirVeri}/{varname}/links/{varname}.{year}.nc')
            ds_oldclim = check_lat_order(ds_oldclim)
            day_window=7
            ds_oldclim = ds_oldclim.drop_vars(['anomaly'])
            ds_oldclim = ds_oldclim.pad(time=day_window,mode='wrap').rolling(time = day_window,center=False,min_periods=day_window).mean()
            ds_oldclim = ds_oldclim.isel(time=slice(day_window,-day_window))
            
            offset_81_10 = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology/2p0.1981-2010/2p0.1981-2010_T2m_gridded.nc'
            offset_91_20 = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology/2p0.1991-2020/2p0.1991-2020_T2m_gridded.nc'
            
            climo_81_10 = check_lat_order(xr.open_dataset(offset_81_10))
            climo_91_20 = check_lat_order(xr.open_dataset(offset_91_20))
            
            
        for i, date in enumerate(dates_in_the_year):
            week3 = date + timedelta(days=21)
            week4 = date + timedelta(days=28)
            # week34_list = [week3, week4, (week3,week4)]# Need to change this part for the week34_Ymd, not working yet
            week34_list = [week3,week4]
            week34_Ymd = [date.strftime('%Y-%m-%d')  for date in week34_list]
           
            # print(date)
            # print(ds_oldclim.sel(time=week34_s))
            # print(ds.sel(time=week34_s))
            # exit
            
            if i%100 == 0: 
                print(f'processing {date}')
            #     print(f'week34 = {week34}')
            
            # Data in the T2m.year.nc files are already 7-day running mean, so we just need to average week3 and 4.

            
            if add_offset:
                if date<dt(2021,5,29):
                    ds_newclim = climo_81_10['climo']
                    print('use 1981-2010 climo')
                else:
                    ds_newclim = climo_91_20['climo']
                    print('use 1991-2020 climo')
                ds_newclim = ds_newclim['doy'].assign_coords(doy=ds_oldclim['doy'])
                # change doy to 1-365
                days = [int(f'{t:%j}') for t in week34_list]
                # print(f'first: {days}')
                days = [min(day, 365) for day in days]
                # change days to 1-365
                # print(f'second: {days}')


                newclim = ds_newclim.sel(doy=days).mean(dim='doy')
                oldclim = ds_oldclim.sel(doy=days).mean(dim=['doy'])['climo']
                # oldclim = ds_oldclim.sel(time=week34_Ymd).sel(doy=days).mean(dim=['time','doy'])
                # sel time and doy to uniquely select the time and doy

                diff = oldclim-newclim
                ds_week34_tmp = ds.sel(time=week34_Ymd).mean(dim='time')
                ds_week34_tmp = ds_week34_tmp+diff
                list_week34_data.append(ds_week34_tmp)
            else:
                # list_week3_data.append (ds.sel(time=week3).drop_vars('time'))
                # list_week4_data.append (ds.sel(time=str(week4)).drop_vars('time'))
                list_week34_data.append(ds.sel(time=week34_Ymd).mean(dim='time'))



        ds_week34= xr.concat(list_week34_data, dim='time')
        ds_week34 = ds_week34.sortby('time') # time is 0-364 here
        ds_week34_mask = ds_week34.where(~mask.isnull())
        
        ds_week34 = xr.Dataset(
            {f'{varname}_wk34':ds_week34_mask,f'{varname}_NorthAmerica_wk34':ds_week34,
            },
            coords={'time':dates_in_the_year,'lat':ds_week34.lat,'lon':ds_week34.lon
            }
        )
        
        # When week3, week4, and week34 are ready, use here
        # ds_week3= xr.concat(list_week3_data, dim='time')
        # ds_week3 = ds_week3.sortby('time') 
        # ds_week3_mask = ds_week3.where(~mask.isnull())
        
        # ds_week4= xr.concat(list_week4_data, dim='time')
        # ds_week4 = ds_week4.sortby('time') 
        # ds_week4_mask = ds_week4.where(~mask.isnull())

        # ds_week34= xr.concat(list_week34_data, dim='time')
        # ds_week34 = ds_week34.sortby('time') # time is 0-364 here
        # ds_week34_mask = ds_week34.where(~mask.isnull())
        
        # ds_week34 = xr.Dataset(
        #     {f'{varname}_wk34':ds_week34_mask,f'{varname}_NorthAmerica_wk34':ds_week34,
        #      f'{varname}_wk3' :ds_week3_mask ,f'{varname}_NorthAmerica_wk3':ds_week3,
        #      f'{varname}_wk4' :ds_week4_mask ,f'{varname}_NorthAmerica_wk4':ds_week4},
        #     coords={'time':dates_in_the_year,'lat':ds_week34.lat,'lon':ds_week34.lon
        #     }
        # )
        # print(ds_week34)
        if add_offset:
            fout = f'{dirVeri}/{varname}/{varname}.{year}.week34_add_offset.nc'
        else:
            # fout = f'{dirVeri}/{varname}/{varname}.{year}.week34.all.nc'
            fout = f'{dirVeri}/{varname}/{varname}.{year}.week34.nc'
        os.system(f'rm -f {fout}')
        ds_week34.to_netcdf(fout)
        
    

        



 




