#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:13:51 2021

@author: slillo

Edited: J.R. Albers 10.4.2022
This function is used to create retrospective (out-of-sample) reforecasts using the NOAA PSL/CPC subseasonal LIM.

- Forecasts are saved as netCDF files via the directories LIMpage_path and FCSTDIR
- LIM forecast operator pickles must already have been created; if they haven't, then read=False must be inserted into the LIMdriver.get_variables() and LIMdriver.get_eofs() calls
- As currently set up, the forecasts use the add_offset flag, which adjusts the reference climatology of the anomalies to be that of the current NOAA CPC base period (currently 1991-2020)

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calendar import monthrange

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib import driver
from lib import data_retrieval
from lib import dataset
from lib import model
from lib import plot
from lib import tools
from lib import verif
from lib.tools import *
# from LIM_CPC import driver
# import data_retrieval
# import LIM_CPC
# from LIM_CPC.tools import *

import warnings
warnings.filterwarnings('ignore')


####################################################################################
### BEGIN USER INPUT ###

#LIMpage_path = f'../Images'
LIMpage_path = f'../Images_retrospective'
FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_beta'
RETROdata_path = './data_retrospective'
getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'

### END USER INPUT ###
####################################################################################

T_START = dt(2020,5,10) #dt(YEAR,MONTH,1)
T_END = dt(2020,6,30) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

####################################################################################
# START RUN CODE
####################################################################################
print('Getting retrospective data:')
dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
                        savetopath=RETROdata_path)
#dataGetter.download(days = [hindcastdays[0]-timedelta(days=7) + timedelta(days=i) for i in range((T_END-T_START).days+7)])
#dataGetter.daily_mean()
dataGetter.download_retrospective(days = [hindcastdays[0]-timedelta(days=7) + timedelta(days=i) for i in range((T_END-T_START).days+7)])
dataGetter.daily_mean_retrospective()

for varname in dataGetter.daily_files.keys():

    os.system(f'rm {dataGetter.savetopath}/{varname}All_TMP.nc')

    ds = nc.Dataset(f'{dataGetter.savetopath}/{varname}All.nc')
    oldtimes = nc.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

    newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname]) if day not in oldtimes]
    #print(newFiles)

    if len(newFiles)>0:

        dss = [xr.open_dataset(f) for f in [f'{dataGetter.savetopath}/{varname}All.nc']+newFiles]

        dstmp = xr.open_dataset(f'{dataGetter.savetopath}/{varname}All.nc')
        lontmp = dstmp['longitude']
        for dstmp in dss:
            dstmp.coords['longitude'] = lontmp
        ds = xr.concat(dss,dim='time').sortby('time')
        print(ds['time'][0], ds['time'][len(ds['time'])-1])
        ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
        os.system(f'rm {dataGetter.savetopath}/{varname}All.nc')
        os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

try:
    os.system(f'rm {dataGetter.savetopath}/*_*')
except:
    pass

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
LIMdriver = driver.Driver(f'namelist_retrospective.py')
LIMdriver.get_variables()
LIMdriver.get_eofs()
LIMdriver.prep_realtime_data(limkey=1)

for T_INIT in hindcastdays:
    START = dt.now()
    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7
    print(weekday)
    try:
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,21+dayoffset,28,28+dayoffset),fullVariance=True)
        if T_INIT<dt(2021,5,29):
            climoffsetfile = 'data_clim/CPC.1981-2010.nc'
        else:
            climoffsetfile = 'data_clim/CPC.1991-2020.nc'    
        LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(21,28),save_to_path=FCSTDIR,add_offset=climoffsetfile)
        if weekday==1 or weekday==4:
            var_name_append = '_Week_34_official_CPC_period_weekday'+str(weekday)
            LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset=climoffsetfile,append_name=var_name_append)
    except:
        print(f'{T_INIT:%Y%m%d} data is unavailable and/or forecast was unable to run')
        pass

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
