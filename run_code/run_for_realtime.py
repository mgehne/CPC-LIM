#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: slillo

Edited: J.R. Albers 10.4.2022
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
from lib.write_html_page import write_month_html, write_day_html
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
pool_Number = 1     # Number of CPU threads that script is allowed to use when saving figure files
credit='NOAA/PSL and University of Colorado/CIRES \nExperimental LIM Forecast (v1.2)'

# Second directory is location where images are copied for posting on PSL website.
# When this code is copied over and made to be the officially running version, the location should be changed to: copy_to_dirs = ['/httpd-test/psd/forecasts/lim_s2s/']
# copy_to_dirs = ['/Projects/jalbers_process/CPC_LIM/10.3.2022_noSetup/lim_s2s/']
copy_to_dirs = ['../lim_s2s/']
page_start_date = dt(2017,1,1)

### END USER INPUT ###
####################################################################################
#%%

####################################################################################
# START RUN CODE
####################################################################################

START = dt.now()

# UPDATE DATA
print('\nGetting realtime data...\n')
t0=dt.now().replace(hour=0,minute=0,second=0,microsecond=0)
dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
                        savetopath=RTdata_path)
dataGetter.download(days = [t0+timedelta(days=i-14) for i in range(14)])

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
        print(dss)
        ds = xr.concat(dss,dim='time').sortby('time')

        ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
        os.system(f'rm {dataGetter.savetopath}/{varname}All.nc')
        os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

try:
    os.system(f'rm {dataGetter.savetopath}/*_*')
except:
    pass

FORECASTDAYS = sorted([t for t in set(sum(dataGetter.available_days.values(),[])) if not os.path.isdir(f'{LIMpage_path}/{t:%Y%m%d}')])


# %%===========================================================================
# INITIALIZE AND RUN BLENDED LIM FORECAST
# =============================================================================

print('\nInitializing...')
LIMdriver = driver.Driver('namelist.py')
LIMdriver.get_variables(read=True)
LIMdriver.get_eofs(read=True)
LIMdriver.prep_realtime_data(limkey=1,verbose=False) #dummy limkey just to get available times
FORECASTDAYS = sorted(list(set(FORECASTDAYS)&set(LIMdriver.RT_VARS['time'])))

#%%
print('\nRunning blended LIM forecasts...')

for T_INIT in FORECASTDAYS:
    START = dt.now()

    dirname = f'{T_INIT:%Y%m%d}'
    FCSTDIR = f'{LIMpage_path}/{dirname}'

    os.system(f'mkdir {FCSTDIR}')

    weekday = T_INIT.weekday()
    dayoffset = (4-weekday)%7

    try:
        print(f'DOING FORECAST FOR {T_INIT:%Y%m%d}')
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=np.arange(0,29+dayoffset),fullVariance=fullVariance,\
                        save_netcdf_path=None)
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

    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_maps,mapLTs)

    def make_loops(varname):
        filenames = [f'{FCSTDIR}/{varname}_lt{l:03}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/{varname}_lt028.png' for l in range(5)]
        os.system('convert -delay 16 -loop 0 '+' '.join(filenames)+f' {FCSTDIR}/{varname}.gif')
        for l in range(0,28,1):
            if l not in (0,14,21,28,21+dayoffset,28+dayoffset):
                os.system(f'rm {FCSTDIR}/{varname}_lt{l:03}.png')
                os.system(f'rm {FCSTDIR}/{varname}-PROB_lt{l:03}.png')

    with mp.Pool(processes=pool_Number) as pool:
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

    try:
        print(f'SAVING FORECAST FOR {T_INIT:%Y%m%d}')
        LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,add_offset='data_clim/CPC.1991-2020.nc')
        LIMdriver.save_netcdf_files(varname='SLP',t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,add_offset='data_clim/SLP.JRA.1991-2020.nc')
        LIMdriver.save_netcdf_files(varname='H500',t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,add_offset='data_clim/H500.JRA.1991-2020.nc')
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,add_offset='data_clim/colIrr.JRA.1991-2020.nc')

        print(f'SAVING CPC PERIOD FORECAST FOR {T_INIT:%Y%m%d}')
        var_name_append = '_Week_34_official_CPC_period'
        LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset='data_clim/CPC.1991-2020.nc',append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SLP',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset='data_clim/SLP.JRA.1991-2020.nc',append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='H500',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset='data_clim/H500.JRA.1991-2020.nc',append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset='data_clim/colIrr.JRA.1991-2020.nc',append_name=var_name_append)

    except:
        print(f'NO FORECAST TO SAVE FOR {T_INIT:%Y%m%d}')
        continue

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run forecast for {T_INIT:%Y%m%d}\n')

    for destination in copy_to_dirs:
        os.system(f'mkdir {destination}{T_INIT:%Y%m}')
        os.system(f'cp -r {FCSTDIR} {destination}{T_INIT:%Y%m}')
        os.system(f'rm {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}/*.nc')
        #add group permissions
        os.system(f'chmod -R g+w {destination}{T_INIT:%Y%m}')


# %%===========================================================================
# Verification for BLENDED LIM
# =============================================================================




VERIFDAYS = [t-timedelta(days=28) for t in FORECASTDAYS]
varname = 'T2m'


for T_INIT_verif in VERIFDAYS:

    try:

        dirname = f'{T_INIT_verif:%Y%m%d}'
        VERIFDIR = f'{LIMpage_path}/{dirname}'
        SKILL_FILE = f'{varname}_skill_all.nc'
        print(f'DOING VERIFICATION FOR {T_INIT_verif:%Y%m%d}')

        getCPCobs([T_INIT_verif+timedelta(days=i) for i in (7,14,21,28)],per=7,savetopath=VERIFDIR)
        getCPCobs(T_INIT_verif+timedelta(days=28),per=14,savetopath=VERIFDIR)


        dirname = f'{T_INIT_verif:%Y%m%d}'
        VERIFDIR = f'{LIMpage_path}/{dirname}'
        skill = make_verif_maps(T_INIT_verif)
        pickle.dump(skill, open( f'{LIMpage_path}/skill_pickles/{T_INIT_verif:%Y%m%d}.p','wb'))

    # MAKE SKILL PLOTS
        dates = [T_INIT_verif+timedelta(days=i) for i in range(-364,1,1)]

        skill_dict = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}

        for T_INIT in dates:
            try:
                skill = pickle.load( open( f'{LIMpage_path}/skill_pickles/{T_INIT:%Y%m%d}.p', 'rb' ) )
                for k,v in skill.items():
                    skill_dict[k].append(v)
            except:
                pass

        #HSS
        fig = plt.figure(figsize=(10,6),dpi=200)
        time = skill_dict['date']
        HSS = skill_dict['HSS']
        HSS_55 = skill_dict['HSS_55']

        HSS_avg = f'{np.nanmean(HSS):0.3f}'
        HSS_55_avg = f'{np.nanmean(HSS_55):0.3f}'

        plt.plot(time,HSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{HSS_avg: >16}')
        plt.plot(time,HSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{HSS_55_avg: >10}')

        plt.yticks(np.arange(-1,1.1,.2))
        xlim = plt.gca().get_xlim()
        plt.plot(xlim,[0,0],'k',linewidth=1.5)
        plt.axis([*xlim,-1.1,1.1])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

        plt.title('Temperature Week-3/4 Heidke Skill Score',fontsize=17)
        plt.xlabel('Initialization Time',fontsize=15)
        plt.ylabel('HSS',fontsize=15)
        plt.legend(loc='lower left',fontsize=10.5)
        plt.grid()
        plt.savefig(f'{VERIFDIR}/{varname}_HSS_timeseries.png',bbox_inches='tight')
        plt.close()

        #RPSS
        fig = plt.figure(figsize=(10,6),dpi=150)
        time = skill_dict['date']
        RPSS = skill_dict['RPSS']
        RPSS_55 = skill_dict['RPSS_55']

        RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
        RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'

        plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{RPSS_avg: >16}')
        plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{RPSS_55_avg: >10}')

        plt.yticks(np.arange(-1,1.1,.2))
        xlim = plt.gca().get_xlim()
        plt.plot(xlim,[0,0],'k',linewidth=1.5)
        plt.axis([*xlim,-1.1,1.1])
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

        plt.title('Temperature Week-3/4 Ranked Probability Skill Score',fontsize=17)
        plt.xlabel('Initialization Time',fontsize=15)
        plt.ylabel('RPSS',fontsize=15)
        plt.legend(loc='lower left',fontsize=10.5)
        plt.grid()
        plt.savefig(f'{VERIFDIR}/{varname}_RPSS_timeseries.png',bbox_inches='tight')
        plt.close()

        for destination in copy_to_dirs:
            os.system(f'rm -r {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}')
            os.system(f'cp -r {VERIFDIR} {destination}{T_INIT_verif:%Y%m}')
            os.system(f'rm {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}/*.nc')

    except:

        print(f'couldnt make verif for {T_INIT_verif}')

# %%===========================================================================
# Update HTML pages
# =============================================================================

# if len(FORECASTDAYS)>0:
#     page_dates = [page_start_date+timedelta(days=i) for i in range(0,(dt.now()-page_start_date).days-1,1)]
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
#         os.system(f'/home/dmwork/WebWorkCommonCode/bin/unAbsSymLnk.sh {destination}index.html')

#         forecastmonths = set([f'{page_start_date+timedelta(d):%Y%m}' for d in range(0,(dt.now()-page_start_date).days-1,1)])
#         webfiles = [f'web_{m}.html' for m in forecastmonths]+[m for m in sorted(forecastmonths)[-3:]]+['index.html']

#         # def publish_files(w):
#         #      os.system(f'/mnt/trio_apps/localpsl/bin/webinstall -l slillo -m "updated LIM page for {max(FORECASTDAYS):%Y%m%d}."  {destination}{w}')

#         # with mp.Pool(mp.cpu_count()) as pool:
#         #     pool.map(publish_files,webfiles)
