#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: slillo

Edited: J.R. Albers 10.4.2022
Edited: Maria Gehne March 2023
Edited: Yuan-Ming Cheng Nov 9 2023

This function is used to create realtime forecasts for the NOAA PSL/CPC subseasonal LIM.

- JRA-55 data is downloaded from the NCAR RDA
- The LIM forecast operator is read in via previously created pickle files
- If new forecast pickles need to be made, 'read=False' must be set in the get_variables and get_eofs calls
- LIM forecast model (and variables) are set via the namelist.py file sent to driver.Driver()
- Blended forecasts from three different LIMs (to account for seasonality) are calculated and figures created from those forecasts; these files are saved in 'LIMpage_path'
- In the plot_map call to LIMdriver, the add_offset flag adjusts the reference climatology of the anomalies to be that of the current NOAA CPC base period (currently 1991-2020)
- Forecasts figures are also copied over to a second directory for posting on the PSL website; these files are saved in 'copy_to_dirs'

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

from lib import driver
from lib import data_retrieval
from lib import write_html_page
from lib.tools import *
from lib.write_html_page import write_month_html, write_day_html

####################################################################################
### BEGIN USER INPUT ###
expt_name = 'realtime'
LIMpage_path = f'<insert_save_path_for_images_here>'
os.system(f'mkdir -p {LIMpage_path}')

RTdata_path = 'data_realtime'
getdataUSER = '0000-0002-6522-4297'
getdataPASS = '645a0323afbe1c1fcc8cdb39f336'
fullVariance = True
DPI=120
pool_Number = 1     # Number of CPU threads that script is allowed to use when saving figure files
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v2.0)'

# Second directory is location where images are copied for posting on PSL website.
# When this code is copied over and made to be the officially running version, the location should be changed to: copy_to_dirs = ['/httpd-test/psd/forecasts/lim_s2s/']
# copy_to_dirs = ['/Projects/jalbers_process/CPC_LIM/10.3.2022_noSetup/lim_s2s/']
# copy_to_dirs = ['../lim_s2s/']
page_start_date = dt(2017,1,1)
os.system(f'mkdir -p {RTdata_path}')
# for destination in copy_to_dirs:
#     os.system(f'mkdir -p {destination}')
### END USER INPUT ###
####################################################################################
#%%

####################################################################################
# START RUN CODE
####################################################################################

# UPDATE DATA
print('\nGetting realtime data...\n')
t0=dt.now().replace(hour=0,minute=0,second=0,microsecond=0)
dataGetter = data_retrieval.getData(orcid_id=getdataUSER,api_token=getdataPASS,\
                        savetopath=RTdata_path)
dataGetter.download(days = [t0+timedelta(days=i-14) for i in range(14)])
dataGetter.daily_mean()

for varname in dataGetter.daily_files.keys():

    os.system(f'rm {dataGetter.savetopath}/{varname}All_TMP.nc')

    if os.path.exists(f'{dataGetter.savetopath}/{varname}All.nc'):

        ds = nc.Dataset(f'{dataGetter.savetopath}/{varname}All.nc')
        oldtimes = nc.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
        newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname]) if day not in oldtimes]
        print(f'previuos {dataGetter.savetopath}/{varname}All.nc exits')
        # print(newFiles)

        if len(newFiles)>0:
            dss = [xr.open_dataset(f) for f in [f'{RTdata_path}/{varname}All.nc']+newFiles]

            dstmp = xr.open_dataset(f'{RTdata_path}/{varname}All.nc') 
            lontmp = dstmp['longitude']
            for dstmp in dss:
                dstmp.coords['longitude'] = lontmp
        else:
            print('No new data appended to the IC files; no need to forecast')
            exit()
    else: 
        print(f'creating new {varname}All.nc')
        newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname])]
        # print(newFiles)

        if len(newFiles)>0:
            dss = [xr.open_dataset(f) for f in newFiles]
            # print(dss)
            dstest = xr.open_dataset(newFiles[0])
            if np.any(dstest['longitude'] < 0):
                lontmp = np.arange(0,360,np.mean(np.diff(dstest['longitude'])))
                # print('new longitude: ',lontmp)
                print('new longitude is created')
            for dstmp in dss:
                dstmp.coords['longitude'] = lontmp
        else:
            print('No new data appended to the IC files; no need to forecast')
            exit()
            
        
    ds = xr.concat(dss,dim='time').sortby('time')

    ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
    os.system(f'rm -f {dataGetter.savetopath}/{varname}All.nc')
    os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

try:
    os.system(f'rm {dataGetter.savetopath}/*_*')
except:
    pass
FORECASTDAYS = sorted([t for t in set(sum(dataGetter.available_days.values(),[])) ])
# # FORECASTDAYS = sorted([t for t in set(sum(dataGetter.available_days.values(),[])) if not os.path.isdir(f'{LIMpage_path}/{t:%Y%m%d}')])
# %%===========================================================================
# INITIALIZE AND RUN BLENDED LIM FORECAST
# =============================================================================
print('\nInitializing...')
LIMdriver = driver.Driver(f'namelist_{expt_name}.py')
LIMdriver.get_variables()
LIMdriver.get_eofs()
LIMdriver.prep_realtime_data(limkey=1,use_sliding_climo_realtime=True, verbose=True) 
FORECASTDAYS = sorted(list(set(FORECASTDAYS)&set(LIMdriver.RT_VARS['time'])))

print(FORECASTDAYS)
print('\nRunning blended LIM forecasts...')

for T_INIT in FORECASTDAYS:
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

    """
    Regression variables. If no regression is needed set this to None, otherwise it specifies the JRA forecast 
    variable (T2m) and the variable we want to have as output. This requires that we have computed the long-term 
    EOFs of the output variable. This can be done by specifying the variable and file locations in the namelist 
    and setting read=False above in the LIMdriver.get_variables and LIMdriver.get_eofs.
    """ 
    pc_convert = None
    # pc_convert = ['T2m','CPCtempHR']

    # Run the LIM forecast
    try:
        print(f'DOING FORECAST FOR {T_INIT:%Y%m%d}')
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=np.arange(0,29+dayoffset),fullVariance=fullVariance,\
                    pc_convert=pc_convert) # Save files using LIMdriver.save_netcdf_files
        # LIMdriver.run_forecast(t_init=T_INIT,lead_times=np.arange(0,29+dayoffset),fullVariance=fullVariance) # Save files using LIMdriver.save_netcdf_files               
    except:
        print(f'NO BLEND FORECAST FOR {T_INIT:%Y%m%d}')
        continue

    # Tvar = 'T2m'# Tvar not used now, commented out.
    if T_INIT<dt(2021,5,29):
        # climoffsetfile = 'data_clim/2p0.1981-2010_T2m.nc'
        climoffsetfile = 'data_clim/2p0.1981-2010'
    else:
        climoffsetfile = 'data_clim/2p0.1991-2020'
    if pc_convert is not None:
        Tvar = pc_convert[1]
    # if Tvar=='CPCtempHR':  
    #     climoffsetfile = 'data_clim/CPCtempHR.day.1991-2020.ltm.nc'

    mapLTs = set([(i,) for i in range(0,29,1)]+[(21,28)]+[(21+dayoffset,),(28+dayoffset,),(21+dayoffset,28+dayoffset)
    ])
    # mapLTs = set([(i,) for i in range(0,3,1)])

    def make_maps(LT):
        LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True, gridded=True,\
                    prop={'interpolate':.25,'levels':np.linspace(-5,5,21),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/T2m')
        # LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,add_offset=None, gridded=True,\
        #             prop={'interpolate':.25,'levels':np.linspace(-5,5,21),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/T2m')

        LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                      add_offset=f'{climoffsetfile}_SLP.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$hPa$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SLP')
        # LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$hPa$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SLP')
        
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                     add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/H500')
        # LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/H500')
        
        LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                   add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                    'levels':(-200,200),'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI,'addtext':credit},\
                    save_to_path = f'{FCSTDIR}/colIrr') 
        # LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':(-200,200),'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI,'addtext':credit},\
        #             save_to_path = f'{FCSTDIR}/no_offset/colIrr') 
              
        LIMdriver.plot_map(varname='SF100',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SF100.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-1e7,1e7,21).astype(int),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-1,1,21)],\
                        'cbar_label':'1e7 $m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SF100')  
        # LIMdriver.plot_map(varname='SF100',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-1e7,1e7,21).astype(int),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-1,1,21)],\
        #                 'cbar_label':'1e7 $m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SF100')  
                  
        LIMdriver.plot_map(varname='SF750',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SF750.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'levels':np.linspace(-1e7,1e7,21).astype(int),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-1,1,21)],\
                        'cbar_label':'1e7 $m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SF750')   
        # LIMdriver.plot_map(varname='SF750',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'levels':np.linspace(-1e7,1e7,21).astype(int),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-1,1,21)],\
        #                 'cbar_label':'1e7 $m^2s^{-1}$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SF750')            
        
        LIMdriver.plot_map(varname='SST'  ,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                    add_offset=f'{climoffsetfile}_SST.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'cmap':{-4:'violet',-3:'mediumblue',-2:'lightskyblue',-1:'w',1:'w',2:'gold',3:'firebrick',4:'violet'},
                          'levels':np.array([-4,-3,-2,-1,0,1,2,3,4]).astype(int),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SST')     
        # LIMdriver.plot_map(varname='SST'  ,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'cmap':{-4:'violet',-3:'mediumblue',-2:'lightskyblue',-1:'w',1:'w',2:'gold',3:'firebrick',4:'violet'},
        #                   'levels':np.array([-4,-3,-2,-1,0,1,2,3,4]).astype(int),'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SST') 

        LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,                 add_offset=f'{climoffsetfile}_SOIL.nc',add_offset_sliding_climo=True,gridded=True,\
                    prop={'interpolate':.25,'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                    'levels':np.linspace(-.4,.4,17),'cbarticks':np.linspace(-.4,.4,9),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-.4,.4,9)],\
                        'dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/SOIL')    
        # LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
        #             prop={'interpolate':.25,'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #             'levels':np.linspace(-.4,.4,17),'cbarticks':np.linspace(-.4,.4,9),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-.4,.4,9)],\
        #                 'dpi':DPI,'addtext':credit},save_to_path = f'{FCSTDIR}/no_offset/SOIL')    

    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_maps,mapLTs)
        

    def make_loops(varname):
        filenames = [f'{FCSTDIR}/{varname}/{varname}_lt{l:03}_{T_INIT:%Y%m%d}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/{varname}/{varname}_lt028_{T_INIT:%Y%m%d}.png' for l in range(5)]
        os.system('convert -delay 16 -loop 0 '+' '.join(filenames)+f' {FCSTDIR}/{varname}.gif')
        # filenames_no_offset = [f'{FCSTDIR}/no_offset/{varname}/{varname}_lt{l:03}_{T_INIT:%Y%m%d}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/no_offset/{varname}/{varname}_lt028_{T_INIT:%Y%m%d}.png' for l in range(5)]
        # os.system('convert -delay 16 -loop 0 '+' '.join(filenames_no_offset)+f' {FCSTDIR}/{varname}_no_offset.gif')
        # for l in range(0,28,1):
            # if l not in (0,14,21,28,21+dayoffset,28+dayoffset):
                # os.system(f'rm {FCSTDIR}/{varname}/{varname}_lt{l:03}_{T_INIT:%Y%m%d}.png')
                # os.system(f'rm {FCSTDIR}/{varname}/{varname}-PROB_lt{l:03}_{T_INIT:%Y%m%d}.png')

    # with mp.Pool(processes=pool_Number) as pool:
    #     pool.map(make_loops,('T2m','SLP','H500','colIrr','SF100','SF750','SST','SOIL'))

    for bounds in [(-15,0),(-7.5,7.5),(0,15)]:
        LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,gridded=True,\
                                prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                                      'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')
        # LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=None,gridded=True,\
        #                         prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
        #                               'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
        #                               save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}_no_offset.png')
        
    for bounds in [(20,40),(30,50),(40,60)]:
        LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,gridded=True,\
                                prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
                                      'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')
        # LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,add_offset=None,gridded=True,\
        #                         prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
        #                               'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
        #                               save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}_no_offset.png')

    LIMdriver.plot_teleconnection(T_INIT=T_INIT,gridded=True,daysback=60,prop={'dpi':DPI},save_to_path = FCSTDIR)

    try:
        print(f'SAVING FORECAST FOR {T_INIT:%Y%m%d}')
        LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/T2m',      add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/SLP',      add_offset=f'{climoffsetfile}_SLP.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/H500',     add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/colIrr',   add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/SF100',    add_offset=f'{climoffsetfile}_SF100.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/SF750',    add_offset=f'{climoffsetfile}_SF750.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/SST',      add_offset=f'{climoffsetfile}_SST.nc',add_offset_sliding_climo=True)
        LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/SOIL',     add_offset=f'{climoffsetfile}_SOIL.nc',add_offset_sliding_climo=True)

        # LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/T2m',    add_offset=None)
        # LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/SLP',    add_offset=None)
        # LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/H500',   add_offset=None)
        # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/colIrr', add_offset=None)
        # LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/SF100',  add_offset=None)
        # LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/SF750',  add_offset=None)
        # LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/SST',    add_offset=None)
        # LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=tuple(range(0,29+dayoffset)),save_to_path=f'{FCSTDIR}/no_offset/SOIL',   add_offset=None)
        
        print(f'SAVING CPC PERIOD FORECAST FOR {T_INIT:%Y%m%d}')
        var_name_append = '_Week_34_official_CPC_period'
        LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/T2m',   add_offset=f'{climoffsetfile}_T2m.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SLP',   add_offset=f'{climoffsetfile}_SLP.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/H500',  add_offset=f'{climoffsetfile}_H500.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/colIrr',add_offset=f'{climoffsetfile}_colIrr.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SF100', add_offset=f'{climoffsetfile}_SF100.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SF750', add_offset=f'{climoffsetfile}_SF750.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SST',   add_offset=f'{climoffsetfile}_SST.nc',add_offset_sliding_climo=True,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/SOIL',  add_offset=f'{climoffsetfile}_SOIL.nc',add_offset_sliding_climo=True,append_name=var_name_append)

        # LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/T2m',     add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SLP',     add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/H500',    add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/colIrr',  add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SF100',   add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SF750',   add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SST',     add_offset=None,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=f'{FCSTDIR}/no_offset/SOIL',    add_offset=None,append_name=var_name_append)

    except:
        print(f'NO FORECAST TO SAVE FOR {T_INIT:%Y%m%d}')
        continue

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run forecast for {T_INIT:%Y%m%d}\n')

    # for destination in copy_to_dirs:
    #     os.system(f'mkdir -p {destination}{T_INIT:%Y%m}')
    #     os.system(f'cp -r {FCSTDIR} {destination}{T_INIT:%Y%m}')
    #     #add group permissions
    #     os.system(f'chmod -R g+w {destination}{T_INIT:%Y%m}')

