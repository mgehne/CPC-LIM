#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: slillo

Edited: J.R. Albers 10.4.2022

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

# Edited import method J.R. Albers 10.4.2022
import lib
from lib import driver
from lib import data_retrieval
from lib import write_html_page
from lib.write_html_page import *
# import LIM_CPC
# from write_html_page import *
# from LIM_CPC import driver
# import data_retrieval

import warnings
warnings.filterwarnings('ignore')

def lat2strNodeg(x):
    deg = u"\u00B0"
    if x<0:
        return f'{abs(x)}S'
    else:
        return f'{x}N'

####################################################################################
### BEGIN USER INPUT ###
LIMpage_path = f'../Images'
copy_to_dirs = ['/Projects/jalbers_process/CPC_LIM/10.3.2022_noSetup/lim_s2s/']
fullVariance = True
gridded = True
DPI=120
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast'
### END USER INPUT ###
####################################################################################



####################################################################################
# START RUN CODE
####################################################################################

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
LIMdriver = driver.Driver('namelist_hindcast.py')
LIMdriver.get_variables()
LIMdriver.get_eofs()

#hindcastdays = [dt(2021,1,1) + timedelta(days=i) for i in range(365)]
hindcastdays = [dt(2022,10,1) + timedelta(days=i) for i in range(19)]

#%%

for T_INIT in hindcastdays:
    START = dt.now()

    # =============================================================================
    # INITIALIZE AND RUN BETA LIM FORECAST WITH FULL VARIANCE
    # =============================================================================

    FCSTDIR = f'{LIMpage_path}/{T_INIT:%Y%m%d}'

    try:
        netcdf_forecasts_saved = len([os.path.join(FCSTDIR, f) for f in os.listdir(FCSTDIR) \
                                      if os.path.isfile(os.path.join(FCSTDIR, f)) and f.endswith('.nc')])
    except:
        netcdf_forecasts_saved = 0

    # if netcdf_forecasts_saved>0:
    #     print(f'ALREADY HAVE NETCDF FILES. SKIPPING FORECAST FOR {T_INIT:%Y%m%d}')
    # else:

    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7

    if True:
        os.system(f'mkdir {FCSTDIR}')

        LIMdriver.prep_realtime_data(limkey=T_INIT.month)

        try:
            print(f'DOING FORECAST FOR {T_INIT:%Y%m%d}')
            LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=np.arange(0,29+dayoffset),fullVariance=fullVariance,\
                            save_netcdf_path=FCSTDIR)
        except:
            print(f'NO BLEND FORECAST FOR {T_INIT:%Y%m%d}')
            continue

        mapLTs = set([(i,) for i in range(0,29,1)]+[(21,28)]+[(21+dayoffset,),(28+dayoffset,),(21+dayoffset,28+dayoffset)])

        def make_maps(LT):
            LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/CPC.1991-2020.nc',gridded=True,\
                        prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = FCSTDIR)
            LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/SLP.JRA.1991-2020.nc',gridded=True,\
                        prop={'levels':np.linspace(-10,10,21).astype(int),'interpolate':1,'cbar_label':'$hPa$','dpi':DPI},save_to_path = FCSTDIR)
            LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/H500.JRA.1991-2020.nc',gridded=True,\
                        prop={'levels':np.linspace(-100,100,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)
            LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/colIrr.JRA.1991-2020.nc',gridded=True,\
                        prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                        'levels':(-200,200),'interpolate':1,'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI},\
                        save_to_path = FCSTDIR)

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(make_maps,mapLTs)

        def make_loops(varname):
            filenames = [f'{FCSTDIR}/{varname}_lt{l:03}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/{varname}_lt028.png' for l in range(5)]
            os.system('convert -delay 16 -loop 0 '+' '.join(filenames)+f' {FCSTDIR}/{varname}.gif')
            for l in range(0,28,1):
                if l not in (0,14,21,28,21+dayoffset,28+dayoffset):
                    os.system(f'rm {FCSTDIR}/{varname}_lt{l:03}.png')
                    os.system(f'rm {FCSTDIR}/{varname}-PROB_lt{l:03}.png')

        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(make_loops,('T2m','SLP','H500','colIrr'))

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

        LIMdriver.plot_teleconnection(T_INIT=T_INIT,gridded=True,daysback=60,prop={'dpi':DPI},save_to_path = FCSTDIR)

        FINISH = dt.now()
        ELAPSED = (FINISH-START).total_seconds()/60
        print(f'\n {ELAPSED:.2f} minutes to run \n FORECAST FOR {T_INIT:%Y%m%d}')


    # =============================================================================
    # Verification
    # =============================================================================


# %%===========================================================================
# Update HTML pages
# =============================================================================

# page_dates = [page_start_date+timedelta(days=i) for i in range((max(FORECASTDAYS)-page_start_date).days,0,-1)]
# for destination in copy_to_dirs:
#     print(f'writing HTML file to {destination}')
#     for T_INIT in page_dates:
#         monthhtml = write_month_html(T_INIT,destination)
#         dayhtml = write_day_html(T_INIT,destination)
#     os.system(f'ln -s {monthhtml} {destination}index.html')


if len(FORECASTDAYS)>0:
    page_dates = [page_start_date+timedelta(days=i) for i in range(0,(dt.now()-page_start_date).days-1,1)]
    for destination in copy_to_dirs:
        print(f'writing HTML file to {destination}')
        for T_INIT in page_dates:
            try:
                monthhtml = write_month_html(T_INIT,destination)
                dayhtml = write_day_html(T_INIT,destination)
            except:
                pass
        os.system(f'unlink {destination}index.html')
        os.system(f'ln -s {monthhtml} {destination}index.html')
        os.system(f'/home/dmwork/WebWorkCommonCode/bin/unAbsSymLnk.sh {destination}index.html')

        forecastmonths = set([f'{page_start_date+timedelta(d):%Y%m}' for d in range(0,(dt.now()-page_start_date).days-1,1)])
        webfiles = [f'web_{m}.html' for m in forecastmonths]+[m for m in sorted(forecastmonths)[-3:]]+['index.html']
        for w in webfiles:
            os.system(f'/mnt/trio_apps/localpsl/bin/webinstall -l slillo -m "updated LIM page for {max(FORECASTDAYS):%Y%m%d}."  {destination}{w}')




#       os.system(f'cp {LIMpage_path}/{varname}_skill.nc {destination}')
#        monthhtml = write_month_html(T_INIT,destination)
#        dayhtml = write_day_html(T_INIT,destination)
#        os.system('ln -s {monthhtml} {destination}index.html')

# =============================================================================
#     if 'time' not in LIMdriver.RT_VARS.keys():
#         LIMdriver.prep_realtime_data(limkey=T_INIT.month)
#
#     T_INIT_verif = T_INIT - timedelta(days=28)
#     VERIFDIR = f'{LIMpage_path}/{T_INIT_verif:%Y%m%d}'
#     varname = 'T2m'
#     print(f'DOING VERIFICATION FOR {T_INIT_verif:%Y%m%d}')
#
#     try:
#         filename = f'{VERIFDIR}/{varname}.{T_INIT_verif:%Y%m%d}.nc'
#         ds = xr.open_dataset(filename)
#         FILE_EXISTS = True
#
#     except:
#         FILE_EXISTS=False
#         print(f'{filename} does not exist')
#
#     skill = {}
#     if FILE_EXISTS:
#         for LT in [(14,),(21,),(28,),(21,28)]:
#
#             ds_sub = ds.sel(time=T_INIT_verif,lead_time=[timedelta(days=l) for l in LT])
#
#             Fmap = np.mean(ds_sub[f'{varname}'].data,axis=0)
#             Emap = np.mean(ds_sub[f'{varname}_spread'].data,axis=0)
#
#             out = LIMdriver.plot_verif(varname=varname,t_init=T_INIT_verif,lead_times=LT,Fmap=Fmap,Emap=Emap,\
#                                 prob_thresh=55,regMask='United States of America',\
#                                 prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$'},\
#                                 save_to_path=f'{LIMpage_path}/{T_INIT_verif:%Y%m%d}')
#             skill[LT] = out
#             ds_sub.close()
#         ds.close()
#
#         ### CONCATENATE SKILL STATS
#         ds = xr.Dataset.from_dict({
#             'coords':{'time':{'dims':('time',),'data':np.array([np.double((T_INIT_verif-dt(1800,1,1)).total_seconds()/3600)]),
#                               'attrs':{'long_name':'Initial time','units':'hours since 1800-01-01 00:00:0.0'}},
#                       },
#             'data_vars': {f'{metric}_lt{"-".join([str(l) for l in lt])}':
#                           {'dims':('time',),'data':[bymetric]}
#                 for lt,bylt in skill.items() for metric,bymetric in bylt.items()},
#             'dims':['time'],
#         })
#         ds.to_netcdf(f'{LIMpage_path}/{varname}_skill_tmp.nc')
#         ds.close()
#
#         try:
#             with xr.open_dataset(f'{LIMpage_path}/{varname}_skill.nc') as ds1,\
#                 xr.open_dataset(f'{LIMpage_path}/{varname}_skill_tmp.nc') as ds2:
#                     ds = xr.concat([ds1,ds2],dim='time').sortby('time')
#                     ds.to_netcdf(f'{LIMpage_path}/{varname}_skill_tmp2.nc')
#             os.system(f'mv {LIMpage_path}/{varname}_skill_tmp2.nc {LIMpage_path}/{varname}_skill.nc')
#             print('saved concatenated skill file')
#         except:
#             os.system(f'mv {LIMpage_path}/{varname}_skill_tmp.nc {LIMpage_path}/{varname}_skill.nc')
#             print('saved new skill file')
#
#         os.system(f'cp -r {VERIFDIR} /httpd-test/psd/forecasts/lim_s2s/')
#         os.system(f'cp {LIMpage_path}/{varname}_skill.nc /httpd-test/psd/forecasts/lim_s2s/')
#
# =============================================================================
