#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: slillo

Edited: J.R. Albers 10.4.2022
This function is used to create realtime forecasts for the beta version of the NOAA PSL/CPC subseasonal LIM (includes soil moisture).

- JRA-55 data is downloaded from the NCAR RDA
- The LIM forecast operator is read in via previously created pickle files
- If new forecast pickles need to be made, 'read=False' must be set in the get_variables and get_eofs calls. Code may error out after pickle creation, if this is the case, set read=True and rerun.
- LIM forecast model (and variables) are set via the namelist.py file sent to driver.Driver()
- Blended forecasts from three different LIMs (to account for seasonality) are calculated and figures created from those forecasts; these files are saved in 'LIMpage_path'
- In the plot_map call to LIMdriver, the add_offset flag adjusts the reference climatology of the anomalies to be that of the current NOAA CPC base period (currently 1991-2020)
- Forecasts figures are also copied over to a second directory for posting on the PSL website; these files are saved in 'copy_to_dirs'

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt,timedelta
import matplotlib.dates as mdates
import xarray as xr
import netCDF4 as nc

# Edited import method J.R. Albers 10.4.2022
import lib
from lib import driver
from lib import data_retrieval
import os
from lib import write_html_page
from lib.write_html_page import *
import multiprocessing as mp
import pickle
from lib import tools
from lib.tools import *
# from LIM_CPC import driver
# import data_retrieval
# import os
# import LIM_CPC
# from write_html_page import *
# import multiprocessing as mp
# import pickle
# from LIM_CPC.tools import *

# Edited import method J.R. ALbers 10.4.2022. Import was formally located just above the 'Verification for BLENDED LIM' code block below
from lib import getCPCobs
from lib.getCPCobs import *
from lib import interp2CPC
from lib.interp2CPC import *
# from getCPCobs import *
# from interp2CPC import *

# Breakpoint packages added JRA 9.29.2022
from pdb import set_trace as bp
import pprint

import warnings
warnings.filterwarnings('ignore')

#%%

####################################################################################
### BEGIN USER INPUT ###

LIMpage_path = '../Images'
RTdata_path = 'data_realtime'
getdataUSER = 'psl.cpc.lim@noaa.gov'
getdataPASS = 're@ltime'
fullVariance = True
DPI=120
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v2.0beta)'

copy_to_dirs = ['./lim_s2s/beta/']
page_start_date = dt(2017,1,1)

### END USER INPUT ###
####################################################################################

####################################################################################
# START RUN CODE
####################################################################################

# START = dt.now()
START = dt(2023,4,23)
# UPDATE DATA
print('\nGetting realtime data...\n')
t0=START.replace(hour=0,minute=0,second=0,microsecond=0)
dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
                        savetopath=RTdata_path)
dataGetter.download(days = [t0+timedelta(days=i-14) for i in range(14)])
#dataGetter.download(days = [dt(2022,12,i) for i in np.arange(22,32,1)])

dataGetter.daily_mean()

for varname in dataGetter.daily_files.keys():

    os.system(f'rm {dataGetter.savetopath}/{varname}All_TMP.nc')

    ds = nc.Dataset(f'{dataGetter.savetopath}/{varname}All.nc')
    oldtimes = nc.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

    newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname]) if day not in oldtimes]
    print(newFiles)

    if len(newFiles)>0:

        dss = [xr.open_dataset(f) for f in [f'data_realtime/{varname}All.nc']+newFiles]

        dstmp = xr.open_dataset(f'data_realtime/{varname}All.nc')
        lontmp = dstmp['longitude']
        for dstmp in dss:
            dstmp.coords['longitude'] = lontmp
        ds = xr.concat(dss,dim='time').sortby('time')

        ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
        os.system(f'rm {dataGetter.savetopath}/{varname}All.nc')
        os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

try:
    os.system(f'rm {dataGetter.savetopath}/*_*')
except:
    pass
# FORECASTDAYS = sorted([dt.now().replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=i) for i in range(7)])
FORECASTDAYS = sorted([START.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=i) for i in range(7)])


# %%===========================================================================
# INITIALIZE AND RUN BLENDED LIM FORECAST
# =============================================================================

print('\nInitializing...')
LIMdriver = driver.Driver('namelist_beta.py')
# LIMdriver.get_variables(read=False,save_netcdf_path = 'data_clim/tmp')
# LIMdriver.get_eofs(read=False,save_netcdf_path='data_clim/EOFs/EOF')
LIMdriver.get_variables(read=True)
LIMdriver.get_eofs(read=True) 
LIMdriver.prep_realtime_data(limkey=1) #dummy limkey just to get available times

FORECASTDAYS = sorted(list(set(FORECASTDAYS)&set(LIMdriver.RT_VARS['time'])))

#%%
print('\nRunning blended LIM forecasts...')
for T_INIT in FORECASTDAYS:
    START = dt.now()

    dirname = f'{T_INIT:%Y%m%d}_BETA'
    FCSTDIR = f'{LIMpage_path}/{dirname}'

    os.system(f'mkdir {FCSTDIR}')

    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7

    try:
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=np.arange(0,29+dayoffset),fullVariance=fullVariance,\
                        save_netcdf_path=FCSTDIR)
    except:
        print(f'NO BLEND FORECAST FOR {T_INIT:%Y%m%d}')
        continue

    #soil_offset = LIMdriver.F_recon[T_INIT]['T2m']*.1
    #LIMdriver.F_recon[T_INIT]['T2m'] = LIMdriver.F_recon[T_INIT]['T2m']+soil_offset

    mapLTs = set([(0,),(14,),(21,),(28,),(21,28)]+[(21+dayoffset,),(28+dayoffset,),(21+dayoffset,28+dayoffset)])

    def make_maps(LT):
        print(f'\n{T_INIT:%Y%m%d} T2m {LT}')
        LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset=None,gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = FCSTDIR)
        print(f'\n{T_INIT:%Y%m%d} SLP {LT}')
        LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/SLP.JRA.1991-2020.nc',gridded=True,\
                    prop={'levels':np.linspace(-10,10,21).astype(int),'cbar_label':'$hPa$','dpi':DPI},save_to_path = FCSTDIR)
        print(f'\n{T_INIT:%Y%m%d} H500 {LT}')
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/H500.JRA.1991-2020.nc',gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)
        print(f'\n{T_INIT:%Y%m%d} colIrr {LT}')
        LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/colIrr.JRA.1991-2020.nc',gridded=True,\
                    prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                    'levels':(-200,200),'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI},\
                    save_to_path = FCSTDIR)
        print(f'\n{T_INIT:%Y%m%d} SOIL {LT}')
        LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                    'levels':np.linspace(-.4,.4,17),'cbarticks':np.linspace(-.4,.4,9),'cbarticklabels':[f'{np.round(i,1):.1f}' for i in np.linspace(-.4,.4,9)],\
                        'interpolate':.25,'dpi':DPI},save_to_path = FCSTDIR)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(make_maps,mapLTs)

    for bounds in [(-15,0),(-7.5,7.5),(0,15)]:
        LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,gridded=True,add_offset='data_clim/colIrr.JRA.1991-2020.nc',\
                                prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                                      'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')

    for bounds in [(20,40),(30,50),(40,60)]:
        LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,gridded=True,add_offset='data_clim/H500.JRA.1991-2020.nc',\
                                prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
                                      'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')


    LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),fullVariance=fullVariance,categories=3,add_offset=None,gridded=True,\
                prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = FCSTDIR)

    LIMdriver.plot_teleconnection(T_INIT=T_INIT,gridded=True,daysback=60,prop={'dpi':DPI},save_to_path = FCSTDIR)

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run \n FORECAST FOR {T_INIT:%Y%m%d}')

    for destination in copy_to_dirs:
        os.system(f'mkdir {destination}{T_INIT:%Y%m}')
        os.system(f'cp -r {FCSTDIR} {destination}{T_INIT:%Y%m}')
        os.system(f'rm -r {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
        os.system(f'mv {destination}{T_INIT:%Y%m}/{dirname} {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
        os.system(f'rm {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}/*.nc')


# %%===========================================================================
# Verification for BETA LIM
# =============================================================================

if False:
    VERIFDAYS = [t-timedelta(days=28) for t in FORECASTDAYS]
    for T_INIT_verif in VERIFDAYS:

        dirname = f'{T_INIT_verif:%Y%m%d}_BETA'
        VERIFDIR = f'{LIMpage_path}/{dirname}'
        varname = 'T2m'
        SKILL_FILE = f'{varname}_skill_all_BETA.nc'
        print(f'DOING VERIFICATION FOR {T_INIT_verif:%Y%m%d}')

        try:
            filename = f'{VERIFDIR}/{varname}.{T_INIT_verif:%Y%m%d}.nc'
            ds = xr.open_dataset(filename)
            FILE_EXISTS = True

        except:
            FILE_EXISTS=False
            print(f'{filename} does not exist')

        skill = {}
        if FILE_EXISTS:
            for LT in [(0,),(14,),(21,),(28,),(21,28)]:# CYM add 0 to output initial conditions

                ds_sub = ds.sel(time=T_INIT_verif,lead_time=[timedelta(days=l) for l in LT])

                Fmap = np.mean(ds_sub[f'{varname}'].data,axis=0)
                Emap = np.mean(ds_sub[f'{varname}_spread'].data,axis=0)

                out = LIMdriver.plot_verif(varname=varname,t_init=T_INIT_verif,lead_times=LT,Fmap=Fmap,Emap=Emap,\
                                    prob_thresh=55,regMask='United States of America',add_offset='data_clim/CPC.1991-2020.nc',\
                                    prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$'},\
                                    save_to_path=VERIFDIR)
                if LT == (21,28):
                    skill[LT] = out
                ds_sub.close()
            ds.close()

            ### CONCATENATE SKILL STATS
            ds = xr.Dataset.from_dict({
                'coords':{'time':{'dims':('time',),'data':np.array([np.double((T_INIT_verif-dt(1800,1,1)).total_seconds()/3600)]),
                                  'attrs':{'long_name':'Initial time','units':'hours since 1800-01-01 00:00:0.0'}},
                          },
                'data_vars': {f'{metric}':
                              {'dims':('time',),'data':[bymetric]}
                    for lt,bylt in skill.items() for metric,bymetric in bylt.items()},
                'dims':['time'],
            })
            ds.to_netcdf(f'{LIMpage_path}/skill_tmp.nc')
            ds.close()

            try:
                with xr.open_dataset(f'{LIMpage_path}/{SKILL_FILE}') as ds1,\
                    xr.open_dataset(f'{LIMpage_path}/skill_tmp.nc') as ds2:
                        ds = xr.concat([ds1,ds2],dim='time').sortby('time')
                        _,index = np.unique(ds['time'],return_index=True)
                        ds = ds.isel(time=index)
                        ds.to_netcdf(f'{LIMpage_path}/skill_tmp2.nc')
                os.system(f'mv {LIMpage_path}/skill_tmp2.nc {LIMpage_path}/{SKILL_FILE}')
                print('saved concatenated skill file')
            except:
                os.system(f'mv {LIMpage_path}/skill_tmp.nc {LIMpage_path}/{SKILL_FILE}')
                print('saved new skill file')

            # MAKE SKILL PLOTS
            with xr.open_dataset(f'{LIMpage_path}/{SKILL_FILE}') as ds:

                ds=ds.sel(time=slice(T_INIT_verif-timedelta(days=365),T_INIT_verif))

                #HSS
                fig = plt.figure(figsize=(10,6),dpi=200)
                time = ds['time']
                HSS = ds['HSS_all'].data
                HSS_55 = ds['HSS_55'].data

                HSS_avg = f'{np.nanmean(HSS):0.3f}'
                HSS_55_avg = f'{np.nanmean(HSS_55):0.3f}'

                plt.plot(time,HSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{HSS_avg: >16}')
                plt.plot(time,HSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{HSS_55_avg: >10}')

                plt.plot([dt(2021,5,28)]*2,[-99,99],color='k',linestyle=(0,(1,1)),linewidth=1.2)
                t = plt.text(dt(2021,5,23,12),1.0,'1981-20210 clim',ha='right',va='center',fontname='Trebuchet MS',fontsize=9.5)
                t.set_bbox(dict(facecolor='0.5',alpha=0.5,edgecolor='0.5'))
                t = plt.text(dt(2021,6,2),1.0,'1991-2020 clim',ha='left',va='center',fontname='Trebuchet MS',fontsize=9.5)
                t.set_bbox(dict(facecolor='0.5',alpha=0.5,edgecolor='0.5'))

                plt.yticks(np.arange(-1,1.1,.2))
                xlim = plt.gca().get_xlim()
                plt.plot(xlim,[0,0],'k',linewidth=1.5)
                plt.axis([*xlim,-1.1,1.1])
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

                plt.title('Temperature Week-3/4 Heidke Skill Score',fontname='Trebuchet MS',fontsize=17)
                plt.xlabel('Initialization Time',fontname='Trebuchet MS',fontsize=15)
                plt.ylabel('HSS',fontname='Trebuchet MS',fontsize=15)
                plt.legend(loc='lower left',fontsize=10.5)
                plt.grid()
                plt.savefig(f'{VERIFDIR}/{varname}_HSS_timeseries.png',bbox_inches='tight')
                plt.close()

                #RPSS
                fig = plt.figure(figsize=(10,6),dpi=150)
                time = ds['time']
                RPSS = ds['RPSS_all'].data
                RPSS_55 = ds['RPSS_55'].data

                RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
                RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'

                plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{RPSS_avg: >16}')
                plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{RPSS_55_avg: >10}')

                plt.plot([dt(2021,5,28)]*2,[-99,99],color='k',linestyle=(0,(1,1)),linewidth=1.2)
                t = plt.text(dt(2021,5,23,12),1.0,'1981-20210 clim',ha='right',va='center',fontname='Trebuchet MS',fontsize=9.5)
                t.set_bbox(dict(facecolor='0.5',alpha=0.5,edgecolor='0.5'))
                t = plt.text(dt(2021,6,2),1.0,'1991-2020 clim',ha='left',va='center',fontname='Trebuchet MS',fontsize=9.5)
                t.set_bbox(dict(facecolor='0.5',alpha=0.5,edgecolor='0.5'))

                plt.yticks(np.arange(-1,1.1,.2))
                xlim = plt.gca().get_xlim()
                plt.plot(xlim,[0,0],'k',linewidth=1.5)
                plt.axis([*xlim,-1.1,1.1])
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

                plt.title('Temperature Week-3/4 Ranked Probability Skill Score',fontsize=17,fontname='Trebuchet MS')
                plt.xlabel('Initialization Time',fontname='Trebuchet MS',fontsize=15)
                plt.ylabel('RPSS',fontname='Trebuchet MS',fontsize=15)
                plt.legend(loc='lower left',fontsize=10.5)
                plt.grid()
                plt.savefig(f'{VERIFDIR}/{varname}_RPSS_timeseries.png',bbox_inches='tight')
                plt.close()

        for destination in copy_to_dirs:
            os.system(f'rm -r {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}')
            os.system(f'cp -r {VERIFDIR} {destination}{T_INIT_verif:%Y%m}')
            os.system(f'mv {destination}{T_INIT_verif:%Y%m}/{dirname} {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}')
            os.system(f'rm {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}/*.nc')



# %%===========================================================================
# Update HTML pages
# =============================================================================
#
# if len(FORECASTDAYS)>0:
#     page_dates = [page_start_date+timedelta(days=i) for i in range(0,(dt.now()-page_start_date).days+1,1)]
#     for destination in copy_to_dirs:
#         print(f'writing HTML file to {destination}')
#         for T_INIT in page_dates:
#             try:
#                 monthhtml = write_month_html(T_INIT,destination)
#                 dayhtml = write_day_html(T_INIT,destination)
#             except:
#                 pass
#         os.system(f'unlink {destination}index.html')
#         os.system(f'ln -s {monthhtml} {destination}index.html')
