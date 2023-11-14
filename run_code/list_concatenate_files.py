import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
from os import listdir
from os.path import isfile, join

available_days={}
T_START = dt(2018,1,1) #dt(YEAR,MONTH,1)
T_END = dt(2022,12,31) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
days = [hindcastdays[0]-timedelta(days=7) + timedelta(days=i) for i in range((T_END-T_START).days+7)]

RETROdata_path = './data_retrospective'
# RT_VARS = ['hgtAll','surfAll','phy2mAll','landAll']
RT_VARS = ['surf']

    
for var in RT_VARS:
    filenames = sorted([join(f'{RETROdata_path}/{var}', f) for f in listdir(f'{RETROdata_path}/{var}') if isfile(join(f'{RETROdata_path}/{var}', f)) and f.endswith('.nc')])
    # I need to mannually mkdir dir of var and move all var*nc to var/ first
    available_days[var]=[]
    for day in days:
        available_days[var].append(day)

    # newFiles = [fname for day,fname in zip(available_days[var],filenames)]
    # print(len(newFiles)); this one doesn't have 2022/12/31
    # print(len(filenames)); this is the full one

    dss = [xr.open_dataset(f) for f in [filenames]]
    for dstmp in dss:
        dstmp.coords['longitude'] = dstmp['longitude']
    # ds = xr.concat(dss,dim='time').sortby('time')
    # print(ds['time'][0], ds['time'][len(ds['time'])-1])
    # os.system(f'rm {RETROdata_path}/{varname}All.nc')
    # ds.to_netcdf(f' {RETROdata_path}/{varname}All.nc')
    