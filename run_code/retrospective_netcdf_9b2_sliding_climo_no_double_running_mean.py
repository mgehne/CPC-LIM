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
import multiprocessing as mp

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
LIMpage_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_retrospective_8_vars_9b2_sliding_climo_no_double_running_mean'
FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_regression'
PLOTDIR = f'{FCSTDIR}/Images_regression/Maps'
# FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_beta'
# PLOTDIR = f'{FCSTDIR}/Images_adjClim/Maps'
RETROdata_path = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/data_retrospective'
getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'
DPI=120
pool_Number = 1     # Number of CPU threads that script is allowed to use when saving figure files
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v1.2)'

### END USER INPUT ###
####################################################################################

T_START = dt(2017,1,1) #dt(YEAR,MONTH,1) 
T_END = dt(2022,12,31) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

####################################################################################
# START RUN CODE
####################################################################################
# print('Getting retrospective data:')
# dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
#                         savetopath=RETROdata_path)
# # CYM: Now we're using reanalysis for restrospective runs, 
# # Use functions download_restrospective and daily_mean_retrospective
# dataGetter.download_retrospective(days = [hindcastdays[0]-timedelta(days=7) + timedelta(days=i) for i in range((T_END-T_START).days+7)])
# dataGetter.daily_mean_retrospective()

  

# for varname in dataGetter.daily_files.keys():

#     # os.system(f'rm {dataGetter.savetopath}/{varname}All_TMP.nc')

#     ds = nc.Dataset(f'{dataGetter.savetopath}/{varname}All.nc')
#     oldtimes = nc.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

#     newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname]) if day not in oldtimes]
#     # Why zipping available_days and daily_files? Aren't they the same?
    
#     if len(newFiles)>0:

#         dss = [xr.open_dataset(f) for f in [f'{dataGetter.savetopath}/{varname}All.nc']+newFiles]

#         dstmp = xr.open_dataset(f'{dataGetter.savetopath}/{varname}All.nc')
#         lontmp = dstmp['longitude']
#         for dstmp in dss:
#             dstmp.coords['longitude'] = lontmp
#         ds = xr.concat(dss,dim='time').sortby('time')
#         print(ds['time'][0], ds['time'][len(ds['time'])-1])
#         ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
#         os.system(f'rm {dataGetter.savetopath}/{varname}All.nc')
#         os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

# try:
#     os.system(f'rm {dataGetter.savetopath}/*_*')
# except:
#     pass

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
LIMdriver = driver.Driver(f'namelist_retrospective_8_vars_9b2_sliding_climo_no_double_running_mean.py')
# LIMdriver = driver.Driver(f'namelist_retrospective_8_vars_test.py')

# LIMdriver.get_variables(read=False,save_netcdf_path = 'data_clim/tmp')
# LIMdriver.get_eofs(read=False,save_netcdf_path='data_clim/EOFs/')
# LIMdriver.get_variables(read=False, save_netcdf_path = 'data_clim/tmp/SOIL_20-74N')
# LIMdriver.get_variables(read=False)
# LIMdriver.get_eofs(read=False)

LIMdriver.get_variables() 
LIMdriver.get_eofs()
LIMdriver.prep_realtime_data(limkey=1)

# pc_convert = ['T2m','CPCtemp']
pc_convert = None

Tvar = 'T2m'
# if pc_convert is None:
#     # climfilebase = 'data_clim/CPC'
#     climfilebase = 'data_clim/CPC.2p0'
    

# elif pc_convert[1]=='CPCtemp':
#     climfilebase = 'data_clim/CPC.2p0'
#     Tvar = pc_convert[1]
# elif pc_convert[1]=='CPCtempHR':
#     climfilebase = 'data_clim/CPC.0p5'  
#     Tvar = pc_convert[1]  

for T_INIT in hindcastdays:
    START = dt.now()
    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7
    #try:
    LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,28),fullVariance=True,pc_convert=pc_convert)
    # LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,21+dayoffset,28,28+dayoffset),fullVariance=False,pc_convert=pc_convert)
    # LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(0,7,14,21,21+dayoffset,28,28+dayoffset),fullVariance=True,pc_convert=pc_convert)
    if T_INIT<dt(2021,5,29):
        # climoffsetfile = climfilebase+'.1981-2010.test.nc'
        # climoffsetfile = climfilebase+'.1981-2010.nc'
        climoffsetfile = 'data_clim/CPC.2p0.1981-2010.CYM.nc' #CYM's 2-deg offset file but it's not the save as the above files, weird 
    else:
        # climoffsetfile = climfilebase+'.1991-2020.test.nc'  
        # climoffsetfile = climfilebase+'.1991-2020.nc' 
        climoffsetfile = 'data_clim/CPC.2p0.1991-2020.CYM.nc' #CYM's 2-deg offset file but it's not the save as the above files, weird 
    print(climoffsetfile)      
    # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(21,28),save_to_path=FCSTDIR,add_offset=None)
    # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(0,7,14,21,28),save_to_path=FCSTDIR,add_offset=None)
    LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}',add_offset=None)
    var_name_append='_offset'
    LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/offset',add_offset=climoffsetfile,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(0,7,14,21,28),save_to_path=FCSTDIR,add_offset=climoffsetfile,append_name=var_name_append)
    # plot maps
    # mapLTs = set([(0,7,14,21,28)])
    mapLTs = set([(21,28)])
    def make_maps(LT):
        LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=True ,pc_convert=pc_convert,add_offset=climoffsetfile,gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{PLOTDIR}/offset')
        LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=True ,pc_convert=pc_convert,add_offset=None,        gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path =f'{PLOTDIR}')  
        # LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=True,pc_convert=pc_convert,add_offset=climoffsetfile,gridded=True,\
                    # prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = PLOTDIR)
    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_maps,mapLTs)
    
    # if weekday==1 or weekday==4:# CYM: this is to offset realtime forecast on Tue/Fri vs CPC Friday forecast
    #     var_name_append = '_Week_34_official_CPC_period_weekday'+str(weekday)
    #     LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset=None,append_name=var_name_append)
    #     # var_name_append = '_Week_34_official_CPC_period_climo_offset_weekday'+str(weekday)
    #     # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset=climoffsetfile,append_name=var_name_append)
    #     # plot maps
    #     mapLTs = set([(21+dayoffset,28+dayoffset)])
    #     def make_maps(LT):
    #         LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=True,pc_convert=pc_convert,add_offset=None,gridded=True,\
    #         # LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=False,pc_convert=pc_convert,add_offset=None,gridded=True,\
    #                     prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = PLOTDIR)
    #         # LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=True,pc_convert=pc_convert,add_offset=climoffsetfile,gridded=True,\
    #         #             prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = PLOTDIR)
    #     with mp.Pool(processes=pool_Number) as pool:
    #         pool.map(make_maps,mapLTs)
#except:
    #    print(f'{T_INIT:%Y%m%d} data is unavailable and/or forecast was unable to run')
    #    pass

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
