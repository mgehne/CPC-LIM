"""_summary_
    This script is the first step to run retrospective/hindcast experiments.
    It downloads JRA data from NCAR RDA in the grib format, 
    calculate daily averages,
    output daily averages as nc files (e.g., hgt_19680101.nc),
    correct longitudes to be all positive,
    output daily files into yearly nc file (e.g., hgt_1968.nc), and
    rm the daily files.
    You would need to use make_rawdata.py next to preprocess the data.
    
    Change Data_path to where you would like to store the data.
    Change year_start and year_end to change the period of data to download
"""
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
# import netCDF4 as nc
import os
# import lib
from lib import data_retrieval

# Data_path="/data/ycheng/JRA/Data/"  # Data have been moved on Jan 11, 2024
Data_path="/Projects/jalbers_process/CPC_LIM/yuan_ming/JRA" 
# year_start = 1958
year_start = 2024
year_end   = 2024
getdataUSER = '0000'
getdataPASS = '0000'

if not os.path.isdir(Data_path):
    print('The directory', Data_path ,'is not present. Creating a new one..')
    os.mkdir(Data_path)
for year in range(year_start,year_end+1,1): 
    print(f'---------We are processing {year}---------')
    T_START = dt(year,1,1) #dt(YEAR,MONTH,1)
    T_END   = dt(year,12,31) #dt(YEAR,MONTH,LASTDAY)
    downloaddays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
    Data_by_year = f'{Data_path}/{year}'
    if not os.path.isdir(Data_by_year):
        print('The directory', Data_by_year ,'is not present. Creating a new one..')
        os.mkdir(Data_by_year)
    dataGetter = data_retrieval.getData(orcid_id=getdataUSER,api_token=getdataPASS,\
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







