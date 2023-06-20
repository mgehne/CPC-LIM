    """_summary_
    This scripts download JRA data from NCAR RDA in the grib format, 
    calculate daily averages,
    output daily averages as nc files (e.g., hgt_19680101.nc),
    correct longitudes to be all positive,
    output daily files into yearly nc file (e.g., hgt_1968.nc), and
    rm the daily files.
    You would use make_rawdata.py next to preprocess the data.

    """
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
# import netCDF4 as nc
import os
# import lib
from lib import data_retrieval

Data_path = f'/data/ycheng/JRA/Data'
# Data_path = f'/scratch/ycheng/JRA/Data'
getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'

if not os.path.isdir(Data_path):
    print('The directory', Data_path ,'is not present. Creating a new one..')
    os.mkdir(Data_path)
for year in range(1958,2023,1): 

    print(f'---------We are processing {year}---------')
    T_START = dt(year,1,1) #dt(YEAR,MONTH,1)
    T_END   = dt(year,12,31) #dt(YEAR,MONTH,LASTDAY)
    downloaddays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
    Data_by_year = f'{Data_path}/{year}'
    if not os.path.isdir(Data_by_year):
        print('The directory', Data_by_year ,'is not present. Creating a new one..')
        os.mkdir(Data_by_year)
    dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
                            savetopath=Data_by_year)

    if year < 2014:# JRA data are montly before 2014
        print('we are before 2013 using download_retrospective_monthly')
        dataGetter.download_retrospective_before_2013(days = downloaddays)
    else:
        print('we are after 2014 using download_retrospective')
        dataGetter.download_retrospective(days = downloaddays)
        
    dataGetter.daily_mean_retrospective()
 
    for varname in dataGetter.daily_files.keys():
        print('-------',varname,'-------')
        newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname])]
        dss = [xr.open_dataset(f) for f in newFiles]
        
        # lonres = dss['longitude'][1]-dss['longitude'][0]
        if any(dss[0]['longitude'] < 0 ):
            print('shifting longitude to postive only values')
            for dstmp in dss:
                dstmp.coords['longitude'] = np.linspace(0, 360, dss[0]['longitude'].shape[0], endpoint=False)
            # print(dss[0]['longitude'])
        ds = xr.concat(dss,dim='time').sortby('time')
        print(ds['time'][0], ds['time'][len(ds['time'])-1])
        ds.to_netcdf(f'{dataGetter.savetopath}/{varname}_{year}.nc')
        os.system(f'rm {dataGetter.savetopath}/{varname}_{year}????.nc')







