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
expt_name = '14_seasonally_varying_vars_hindcast_fold_9'
LIMpage_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_retrospective_{expt_name}'
# FCSTDIR = f'{LIMpage_path}'
os.system(f'mkdir -p {LIMpage_path}')

getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'
fullVariance = True
DPI=120
pool_Number = 1     # Number of CPU threads that script is allowed to use when saving figure files
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v2)'

### END USER INPUT ###
####################################################################################

T_START = dt(2005,1,1) #dt(YEAR,MONTH,1) 
# T_START = dt(2017,11,29) #dt(YEAR,MONTH,1) 
T_END = dt(2010,12,31) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

####################################################################################
# START RUN CODE
####################################################################################

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
LIMdriver = driver.Driver(f'namelist_{expt_name}.py')
LIMdriver.get_variables(read=False) 
LIMdriver.get_eofs(read=False)
LIMdriver.prep_realtime_data(limkey=1)

# pc_convert = ['T2m','CPCtemp']
pc_convert = None

Tvar = 'T2m'

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
                    pc_convert=pc_convert) # Save files using LIMdriver.save_netcdf_files
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

        # LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                      add_offset=f'{climoffsetfile}_SLP.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$hPa$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SLP')
        # LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$hPa$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SLP')
        
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                     add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/H500')
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/H500')
        
        # LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                   add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':(-200,200),'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI,'addtext':credit},\
        #             save_to_path = f'{FCSTDIR}/colIrr') 
        # LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':(-200,200),'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI,'addtext':credit},\
        #             save_to_path = f'{FCSTDIR}/no_offset/colIrr') 
              
        # LIMdriver.plot_map(varname='SF100',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SF100.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'levels':np.linspace(-100e5,100e5,21).astype(int),'cbar_label':'$m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SF100')  
        # LIMdriver.plot_map(varname='SF100',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-100e5,100e5,21).astype(int),'cbar_label':'$m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SF100')  
                  
        # LIMdriver.plot_map(varname='SF750',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SF750.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'levels':np.linspace(-100e5,100e5,21).astype(int),'cbar_label':'$m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SF750')   
        # LIMdriver.plot_map(varname='SF750',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-100e5,100e5,21).astype(int),'cbar_label':'$m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SF750')            
        
        # LIMdriver.plot_map(varname='SST'  ,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SST.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SST')     
        # LIMdriver.plot_map(varname='SST'  ,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SST') 

        # LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                 add_offset=f'{climoffsetfile}_SOIL.nc',add_offset_sliding_climo=True,gridded=True,\
        #             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':np.linspace(-.4,.4,17),'cbarticks':np.linspace(-.4,.4,9),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-.4,.4,9)],\
        #                 'dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SOIL')    
        # LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':np.linspace(-.4,.4,17),'cbarticks':np.linspace(-.4,.4,9),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-.4,.4,9)],\
        #                 'dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SOIL')    

    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_maps,mapLTs)
        

    # def make_loops(varname):
    #     filenames = [f'{FCSTDIR}/{varname}/{varname}_lt{l:03}_{T_INIT:%Y%m%d}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/{varname}/{varname}_lt028_{T_INIT:%Y%m%d}.png' for l in range(5)]
    #     os.system('convert -delay 16 -loop 0 '+' '.join(filenames)+f' {FCSTDIR}/{varname}.gif')
    #     filenames_no_offset = [f'{FCSTDIR}/no_offset/{varname}/{varname}_lt{l:03}_{T_INIT:%Y%m%d}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/no_offset/{varname}/{varname}_lt028_{T_INIT:%Y%m%d}.png' for l in range(5)]
    #     os.system('convert -delay 16 -loop 0 '+' '.join(filenames_no_offset)+f' {FCSTDIR}/{varname}_no_offset.gif')

    # with mp.Pool(processes=pool_Number) as pool:
    #     # pool.map(make_loops,('T2m','SLP','H500','colIrr','SF100','SF750','SST','SOIL'))
    #     pool.map(make_loops,('T2m','H500','colIrr'))

    # for bounds in [(-15,0),(-7.5,7.5),(0,15)]:
    #     LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,gridded=True,\
    #                             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
    #                                   'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
    #                                   save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')
    #     LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=None,gridded=True,\
    #                             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
    #                                   'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
    #                                   save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}_no_offset.png')
        
    # for bounds in [(20,40),(30,50),(40,60)]:
    #     LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,gridded=True,\
    #                             prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
    #                                   'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
    #                                   save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')
    #     LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=None,gridded=True,\
    #                             prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
    #                                   'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
    #                                   save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}_no_offset.png')

    # LIMdriver.plot_teleconnection(T_INIT=T_INIT,gridded=True,daysback=60,prop={'dpi':DPI},save_to_path = FCSTDIR)

    print(f'SAVING FORECAST FOR {T_INIT:%Y%m%d}')
    LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/T2m',      add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True)
    LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/H500',     add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True)
    # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/colIrr',   add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True)
    
    LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/T2m',    add_offset=None)
    LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/H500',   add_offset=None)
    # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(21,28),save_to_path=f'{FCSTDIR}/no_offset/colIrr', add_offset=None)
   
    # print(f'SAVING CPC PERIOD FORECAST FOR {T_INIT:%Y%m%d}')
    # var_name_append = '_Week_34_official_CPC_period'
    # LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/T2m',   add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SLP',   add_offset=f'{climoffsetfile}_SLP.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/H500',  add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/colIrr',add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SF100', add_offset=f'{climoffsetfile}_SF100.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SF750', add_offset=f'{climoffsetfile}_SF750.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SST',   add_offset=f'{climoffsetfile}_SST.nc',add_offset_sliding_climo=True,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SOIL',  add_offset=f'{climoffsetfile}_SOIL.nc',add_offset_sliding_climo=True,append_name=var_name_append)

    # LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/T2m',     add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SLP',     add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/H500',    add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/colIrr',  add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SF100',   add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SF750',   add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SST',     add_offset=None,append_name=var_name_append)
    # LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SOIL',    add_offset=None,append_name=var_name_append)


    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
