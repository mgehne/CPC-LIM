#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:13:51 2021

@author: slillo

Edited: J.R. Albers 10.4.2022
Edited: Maria Gehne March 2023
Edited: Yuan-Ming Cheng Nov 9 2023

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
expt_name = 'v2p0_dev_output_C0_Ctau_reforecast'
LIMpage_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_{expt_name}'
os.system(f'mkdir -p {LIMpage_path}')

getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'
fullVariance = True
DPI=120
pool_Number = 1     # Number of CPU threads that script is allowed to use when saving figure files
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v2.0)'

### END USER INPUT ###
####################################################################################

T_START = dt(2017,1,1) #dt(YEAR,MONTH,1) 
T_END = dt(2017,12,31) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

####################################################################################
# START RUN CODE
####################################################################################

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
LIMdriver = driver.Driver(f'namelist_{expt_name}.py')
LIMdriver.get_variables(read=True) 
LIMdriver.get_eofs(read=True)
LIMdriver.prep_realtime_data(limkey=1)

# pc_convert = ['T2m','CPCtemp']
pc_convert = None

Tvar = 'T2m'
os.system(f'mkdir -p {LIMpage_path}/model')

for T_INIT in hindcastdays:
    START = dt.now()
    
    dirname = f'{T_INIT:%Y%m%d}'
    FCSTDIR = f'{LIMpage_path}/{dirname}'

    os.system(f'mkdir -p {FCSTDIR}')
    os.system(f'mkdir -p {FCSTDIR}/no_offset')
    for key in LIMdriver.RT_VARS:
        os.system(f'mkdir -p {FCSTDIR}/{key}')
        os.system(f'mkdir -p {FCSTDIR}/no_offset/{key}')
    
    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7
    try:
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,28),fullVariance=fullVariance,\
                    pc_convert=pc_convert,save_netcdf_path=f'{LIMpage_path}/model') # Save files using LIMdriver.save_netcdf_files
    except:
        print(f'NO BLEND FORECAST FOR {T_INIT:%Y%m%d}')
        continue

    if T_INIT<dt(2021,5,29):
        climoffsetfile = 'data_clim/2p0.1981-2010'
    else:
        climoffsetfile = 'data_clim/2p0.1991-2020'
    if pc_convert is not None:
        Tvar = pc_convert[1]
        
    print(climoffsetfile)      
    # plot maps
    mapLTs = set([(21,28)])

    def make_maps(LT):
        LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True, gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/T2m')
        LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,add_offset=None, gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/T2m')


        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                     add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/H500')
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/H500')
        
  
    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_maps,mapLTs)
        

 
    print(f'SAVING FORECAST FOR {T_INIT:%Y%m%d}')
    LIMdriver.save_netcdf_files(varname='T2m',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/T2m',        add_offset=f'{climoffsetfile}_T2m.nc',      add_offset_sliding_climo=True)
    LIMdriver.save_netcdf_files(varname='H500',     t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/H500',       add_offset=f'{climoffsetfile}_H500.nc',     add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='colIrr',   t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/colIrr',     add_offset=f'{climoffsetfile}_colIrr.nc',   add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='SF750',    t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/SF750',      add_offset=f'{climoffsetfile}_SF750.nc',    add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='SF100',    t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/SF100',      add_offset=f'{climoffsetfile}_SF100.nc',    add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='SST',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/SST',        add_offset=f'{climoffsetfile}_SST.nc',      add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='SOIL',     t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/SOIL',       add_offset=f'{climoffsetfile}_SOIL.nc',     add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='SLP',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/SLP',        add_offset=f'{climoffsetfile}_SLP.nc',      add_offset_sliding_climo=True)
    
    LIMdriver.save_netcdf_files(varname='T2m',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/T2m',      add_offset=None)
    LIMdriver.save_netcdf_files(varname='H500',     t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/H500',     add_offset=None)
    # LIMdriver.save_netcdf_files(varname='colIrr',   t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/colIrr',   add_offset=None)
    # LIMdriver.save_netcdf_files(varname='SF750',    t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/SF750',    add_offset=None)
    # LIMdriver.save_netcdf_files(varname='SF100',    t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/SF100',    add_offset=None)
    # LIMdriver.save_netcdf_files(varname='SST',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/SST',      add_offset=None)
    # LIMdriver.save_netcdf_files(varname='SOIL',     t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/SOIL',     add_offset=None)
    # LIMdriver.save_netcdf_files(varname='SLP',      t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/SLP',      add_offset=None)
   

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
