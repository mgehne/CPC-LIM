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
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import traceback

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
from lib.dataset import varDataset

# Edited import method J.R. ALbers 10.4.2022. Import was formally located just above the 'Verification for BLENDED LIM' code block below
from lib import getCPCobs
from lib.getCPCobs import *
# from lib import interp2CPC
# from lib.interp2CPC import *
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
expt_name = '10d_sliding_climo'
LIMpage_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_realtime_{expt_name}'
os.system(f'mkdir -p {LIMpage_path}')


# RTdata_path = 'data_realtime'
RTdata_path = 'data_test_realtime'
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
os.system(f'mkdir -p {RTdata_path}')
### END USER INPUT ###
####################################################################################
#%%

####################################################################################
# START RUN CODE
####################################################################################

START = dt.now()

# UPDATE DATA
# print('\nGetting realtime data...\n')
t0=dt(2023,1,15)
# t0=dt.now().replace(hour=0,minute=0,second=0,microsecond=0)
# dataGetter = data_retrieval.getData(email=getdataUSER,password=getdataPASS,\
#                         savetopath=RTdata_path)
# dataGetter.download(days = [t0+timedelta(days=i-14) for i in range(14)])
# #dataGetter.download(days = [dt(2022,12,i) for i in np.arange(22,32,1)])

# dataGetter.daily_mean()

# for varname in dataGetter.daily_files.keys():

#     os.system(f'rm {dataGetter.savetopath}/{varname}All_TMP.nc')

#     ds = nc.Dataset(f'{dataGetter.savetopath}/{varname}All.nc')
#     oldtimes = nc.num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

#     newFiles = [fname for day,fname in zip(dataGetter.available_days[varname],dataGetter.daily_files[varname]) if day not in oldtimes]
#     print(newFiles)

#     if len(newFiles)>0:

#         dss = [xr.open_dataset(f) for f in [f'data_realtime/{varname}All.nc']+newFiles]

#         dstmp = xr.open_dataset(f'data_realtime/{varname}All.nc')
#         lontmp = dstmp['longitude']
#         for dstmp in dss:
#             dstmp.coords['longitude'] = lontmp
#         ds = xr.concat(dss,dim='time').sortby('time')

#         ds.to_netcdf(f'{dataGetter.savetopath}/{varname}All_TMP.nc')
#         os.system(f'rm {dataGetter.savetopath}/{varname}All.nc')
#         os.system(f'mv {dataGetter.savetopath}/{varname}All_TMP.nc {dataGetter.savetopath}/{varname}All.nc')

# try:
#     os.system(f'rm {dataGetter.savetopath}/*_*')
# except:
#     pass
# FORECASTDAYS = sorted([t for t in set(sum(dataGetter.available_days.values(),[])) if not os.path.isdir(f'{LIMpage_path}/{t:%Y%m%d}')])
# FORECASTDAYS = [t0+timedelta(days=i-14) for i in range(14)]
T_START = dt(2023,7,28) #dt(YEAR,MONTH,1)
T_END = dt(2023,8,12) #dt(YEAR,MONTH,LASTDAY)
FORECASTDAYS = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
# print(FORECASTDAYS)
# %%===========================================================================
# INITIALIZE AND RUN BLENDED LIM FORECAST
# =============================================================================
# print('\nInitializing...')
LIMdriver = driver.Driver(f'namelist_{expt_name}.py')
# LIMdriver.get_variables(read=False)
# LIMdriver.get_eofs(read=False,save_netcdf_path=None)
LIMdriver.get_variables()
LIMdriver.get_eofs()

# for varname in ['T2m']:
# for name in LIMdriver.use_vars.keys():
#     dstmp   = xr.open_dataset(f'data_realtime/surfAll.nc')
#     dstmp   = dstmp.sel(time=slice(f'{t0.year -1 }-12-01', None)) # select starting from past year to avoid problems with running mean
#     # dstmp.sel[time=dstmp['time'].year = f'{t0.year}']
#     print(dstmp['time'])
#     dsclimo = xr.open_dataset(f'data_clim/{varname}/{varname}.{t0.year-1}.nc')
#     sliding_climo = np.array(dsclimo['climo'][:])
#     print(sliding_climo.shape)
#     # varobj = LIMdriver.use_vars[varname]['data']
#     # sliding_climo = np.array([varobj.flatten(i) for i in climo])
#     # sliding_climo[abs(sliding_climo)>1e29]=np.nan
#     climo_dict = {'climo':sliding_climo}
#     LIMdriver.RT_VARS[varname]['info'][2].update(climo_dict)
#     out = varDataset('t2m',*LIMdriver.RT_VARS[varname]['info'][:-1],**LIMdriver.RT_VARS[varname]['info'][-1])
#     # anomaly = get_anomaly(dstmp[varname],ds['time'],sliding_climo)
#     # running_mean = get_running_mean(anomaly,LIMdriver.time_window)[LIMdriver.time_window:]
#     # os.system(f'rm -f ./t2m.????.nc')
#     os.system(f'rm -f ./t2m.all.nc')
#     out.save_to_netcdf('./')

#     # dstmp['time'] = dstmp['time'][LIMdriver.time_window:]

LIMdriver.prep_realtime_data(limkey=1,use_sliding_climo_realtime=True, verbose=True) #dummy limkey just to get available times
# LIMdriver.prep_realtime_data(limkey=1, verbose=True) #dummy limkey just to get available times
# FORECASTDAYS = sorted(list(set(FORECASTDAYS)&set(LIMdriver.RT_VARS['time'])))
#%%
print('\nRunning blended LIM forecasts...')

for T_INIT in FORECASTDAYS:
    START = dt.now()

    dirname = f'{T_INIT:%Y%m%d}'
    FCSTDIR = f'{LIMpage_path}/{dirname}'

    os.system(f'mkdir {FCSTDIR}')

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
                    pc_convert=pc_convert,save_netcdf_path=FCSTDIR)                
    except:
        print(f'NO BLEND FORECAST FOR {T_INIT:%Y%m%d}')
        continue

    Tvar = 'T2m'
    # tclim_file = 'data_clim/CPC.temp.1991-2020.nc'
    # tclim_file = 'data_clim/CPC.1991-2020.nc'# Sam and Maria's offset file
    tclim_file = 'data_clim/CPC.2p0.1991-2020.CYM.nc' #CYM's 2-deg offset file
    if pc_convert is not None:
        Tvar = pc_convert[1]
    if Tvar=='CPCtempHR':  
        tclim_file = 'data_clim/CPCtempHR.day.1991-2020.ltm.nc'

    mapLTs = set([(i,) for i in range(0,29,1)]+[(21,28)]+[(21+dayoffset,),(28+dayoffset,),(21+dayoffset,28+dayoffset)
    ])
    print('before make_maps')
    def make_maps(LT):
        LIMdriver.plot_map(varname='T2m',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,add_offset=tclim_file,add_offset_sliding_climo=True, gridded=True,\
        # LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,pc_convert=pc_convert,gridded=True,\
                    prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = FCSTDIR)
  #      LIMdriver.plot_map(varname=Tvar,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,add_offset='data_clim/CPC.temp.1991-2020.nc',gridded=True,\
  #                  prop={'levels':np.linspace(-5,5,21),'interpolate':.25,'cbar_label':'$^oC$','dpi':DPI,'addtext':credit},save_to_path = FCSTDIR)
        # All off_set are off except for T2m
        LIMdriver.plot_map(varname='SLP',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'levels':np.linspace(-10,10,21).astype(int),'interpolate':1,'cbar_label':'$hPa$','dpi':DPI},save_to_path = FCSTDIR)
        LIMdriver.plot_map(varname='H500',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'levels':np.linspace(-100,100,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)
        LIMdriver.plot_map(varname='colIrr',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                    'levels':(-200,200),'interpolate':1,'cbar_label':'$W/m^2$','figsize':(10,3.5),'drawstates':False,'latlon':True,'central_longitude':180,'dpi':DPI},\
                    save_to_path = FCSTDIR)
        print('before make_maps of SF100')
        
        LIMdriver.plot_map(varname='SF100',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'levels':np.linspace(-300e5,300e5,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)        
        LIMdriver.plot_map(varname='SF750',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'levels':np.linspace(-100e5,100e5,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)     
        print('before make_maps of SST')
        
        LIMdriver.plot_map(varname='SST'  ,t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
                    prop={'levels':np.linspace(-10,10,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)     
        # LIMdriver.plot_map(varname='SOIL',t_init=T_INIT,lead_times=LT,fullVariance=fullVariance,gridded=True,\
        #             prop={'levels':np.linspace(-0.5,0.5,21).astype(int),'interpolate':1,'cbar_label':'$m$','dpi':DPI},save_to_path = FCSTDIR)    
    print('before with mp.Pool')
        
    with mp.Pool(processes=pool_Number) as pool:
        print('before with mp.Pool make_maps')
        pool.map(make_maps,mapLTs)
        print('after with mp.Pool make_maps')
        

    def make_loops(varname):
        filenames = [f'{FCSTDIR}/{varname}_lt{l:03}.png' for l in range(0,28,1)]+[f'{FCSTDIR}/{varname}_lt028.png' for l in range(5)]
        os.system('convert -delay 16 -loop 0 '+' '.join(filenames)+f' {FCSTDIR}/{varname}.gif')
        for l in range(0,28,1):
            if l not in (0,14,21,28,21+dayoffset,28+dayoffset):
                os.system(f'rm {FCSTDIR}/{varname}_lt{l:03}.png')
                os.system(f'rm {FCSTDIR}/{varname}-PROB_lt{l:03}.png')

    with mp.Pool(processes=pool_Number) as pool:
        pool.map(make_loops,(Tvar,'SLP','H500','colIrr'))

    for bounds in [(-15,0),(-7.5,7.5),(0,15)]:
        LIMdriver.plot_timelon(varname='colIrr',t_init=T_INIT,lat_bounds=bounds,daysback=75,gridded=True,\
                                prop={'cmap':{-2:'darkorange',-1:'sienna',-0.2:'w',0.2:'w',1:'seagreen',2:'turquoise'},\
                                      'levels':(-200,200),'cbar_label':'$W/m^2$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_trop_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')

    for bounds in [(20,40),(30,50),(40,60)]:
        LIMdriver.plot_timelon(varname='H500',t_init=T_INIT,lat_bounds=bounds,daysback=75,gridded=True,\
                                prop={'cmap':{-2:'darkorchid',-1:'dodgerblue',-0.2:'w',0.2:'w',1:'tomato',2:'firebrick'},\
                                      'levels':(-100,100),'cbar_label':'$m$','dpi':DPI},\
                                      save_to_file=f'{FCSTDIR}/HOV_500_{lat2strNodeg(bounds[0])}{lat2strNodeg(bounds[1])}.png')

    LIMdriver.plot_teleconnection(T_INIT=T_INIT,gridded=True,daysback=60,prop={'dpi':DPI},save_to_path = FCSTDIR)

    try:
        print(f'SAVING FORECAST FOR {T_INIT:%Y%m%d}')
        var_name_append = '_offset'
        LIMdriver.save_netcdf_files(varname='T2m'   ,t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,add_offset=tclim_file,add_offset_sliding_climo=True,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(0,14,21,28),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        
        print(f'SAVING CPC PERIOD FORECAST FOR {T_INIT:%Y%m%d}')
        var_name_append = '_Week_34_official_CPC_period'
        LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,add_offset=tclim_file,add_offset_sliding_climo=True,append_name=var_name_append)
        # LIMdriver.save_netcdf_files(varname=Tvar,t_init=T_INIT,lead_times=(0+dayoffset,14+dayoffset,21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SLP'   ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='H500'  ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='colIrr',t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF100' ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SF750' ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SST'   ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)
        LIMdriver.save_netcdf_files(varname='SOIL'  ,t_init=T_INIT,lead_times=(21+dayoffset,28+dayoffset),save_to_path=FCSTDIR,append_name=var_name_append)

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

exit()
# %%===========================================================================
# Verification for BLENDED LIM
# =============================================================================
from lib import interp2CPC
from lib.interp2CPC import *
#VERIFDAYS = [dt(2023,1,i) for i in np.arange(1,10,1)]
VERIFDAYS = [t-timedelta(days=28) for t in FORECASTDAYS]
varname = 'T2m'

print(FORECASTDAYS)
print(VERIFDAYS)

for T_INIT_verif in VERIFDAYS:

    weekday = T_INIT_verif.weekday()
    dayoffset = (4-weekday)%7

    try:

        dirname = f'{T_INIT_verif:%Y%m%d}'
        VERIFDIR = f'{LIMpage_path}/{dirname}'

        print(f'DOING VERIFICATION FOR {T_INIT_verif:%Y%m%d}')#

        getCPCobs([T_INIT_verif+timedelta(days=i) for i in (7,14,21,28)],per=7,savetopath=VERIFDIR)
        getCPCobs(T_INIT_verif+timedelta(days=28),per=14,savetopath=VERIFDIR)

        print('make verification maps and skill scores')
        dirname = f'{T_INIT_verif:%Y%m%d}'
        VERIFDIR = f'{LIMpage_path}/{dirname}'
        try:
            skill = make_verif_maps(T_INIT_verif,VERIFDIR)
            pickle.dump(skill, open( f'{LIMpage_path}/skill_pickles/{T_INIT_verif:%Y%m%d}.p','wb'))
            ds = xr.Dataset(skill)
            ds.to_netcdf(f'{LIMpage_path}/skill_pickles/{T_INIT_verif:%Y%m%d}.nc')
            ds.close()
        except:    
            pass
        try:
            skill = make_verif_maps_CPCperiod(T_INIT_verif,VERIFDIR,dayoffset)
            pickle.dump(skill, open( f'{LIMpage_path}/skill_pickles/{T_INIT_verif:%Y%m%d}.CPCperiod.p','wb'))
            ds = xr.Dataset(skill)
            ds.to_netcdf(f'{LIMpage_path}/skill_pickles/{T_INIT_verif:%Y%m%d}.CPCperiod.nc')
            ds.close()
        except:
            pass

    # MAKE SKILL PLOTS
        dates = [T_INIT_verif+timedelta(days=i) for i in range(-364,1,1)]

        skill_dict = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}
        skill_dict_CPC = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}

        for T_INIT in dates:
            try:
                skill = pickle.load( open( f'{LIMpage_path}/skill_pickles/{T_INIT:%Y%m%d}.p', 'rb' ) )
                for k,v in skill.items():
                    skill_dict[k].append(v)
                skill = pickle.load( open( f'{LIMpage_path}/skill_pickles/{T_INIT:%Y%m%d}.CPCperiod.p', 'rb' ) )
                for k,v in skill.items():
                    skill_dict_CPC[k].append(v)    
            except:
                pass

        #HSS
        fig = plt.figure(figsize=(10,6),dpi=200)
        time = skill_dict['date']
        HSS = skill_dict['HSS']
        HSS_55 = skill_dict['HSS_55']
        time_CPC = skill_dict_CPC['date']
        HSS_CPC = skill_dict_CPC['HSS']
        HSS_55_CPC = skill_dict_CPC['HSS_55']

        HSS_avg = f'{np.nanmean(HSS):0.3f}'
        HSS_55_avg = f'{np.nanmean(HSS_55):0.3f}'
        HSS_CPC_avg = f'{np.nanmean(HSS_CPC):0.3f}'
        HSS_55_CPC_avg = f'{np.nanmean(HSS_55_CPC):0.3f}'

        plt.plot(time,HSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{HSS_avg: >16}')
        plt.plot(time,HSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{HSS_55_avg: >10}')
        plt.plot(time_CPC,HSS_CPC,color='dodgerblue',linestyle='dashed',label=f'{"CPC period CONUS": <12}'+f'{HSS_CPC_avg: >16}')
        plt.plot(time_CPC,HSS_55_CPC,color='darkorange',linestyle='dashed',label=f'{"CPC period CONUS >55%": <12}'+f'{HSS_55_CPC_avg: >10}')

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
        RPSS_CPC = skill_dict_CPC['RPSS']
        RPSS_55_CPC = skill_dict_CPC['RPSS_55']

        RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
        RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'
        RPSS_CPC_avg = f'{np.nanmean(RPSS_CPC):0.3f}'
        RPSS_55_CPC_avg = f'{np.nanmean(RPSS_55_CPC):0.3f}'

        plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{RPSS_avg: >16}')
        plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{RPSS_55_avg: >10}')
        plt.plot(time_CPC,RPSS_CPC,color='dodgerblue',linestyle='dashed',label=f'{"CPC period CONUS": <12}'+f'{RPSS_CPC_avg: >16}')
        plt.plot(time_CPC,RPSS_55_CPC,color='darkorange',linestyle='dashed',label=f'{"CPC period CONUS >55%": <12}'+f'{RPSS_55_CPC_avg: >10}')

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
        print(traceback.format_exc())
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
