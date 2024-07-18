import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../') # This allows import of lib that is one level up
from lib.tools import *
import warnings
import scipy.io


expt_name="fixed_58-16_climo"
# expt_name="v2p0"

forecast_periods_input = {
"reforecast"     :  (2017,2022),
# "hindcast_fold_10": (2011,2016),
# "hindcast_fold_9" : (2005,2010),    
# "hindcast_fold_8" : (1999,2004),    
# "hindcast_fold_7" : (1993,1998),
# "reforecast"     :  (2017,2022),
# "hindcast_fold_10": (2011,2016),
# "hindcast_fold_9" : (2005,2010),    
# "hindcast_fold_8" : (1999,2004),    
# "hindcast_fold_7" : (1993,1998),    
# "hindcast_fold_6" : (1987,1992),    
# "hindcast_fold_5" : (1981,1986),    
# "hindcast_fold_4" : (1979,1980),    
# "hindcast_fold_4" : (1975,1979),    
# "hindcast_fold_3" : (1969,1974),    
# "hindcast_fold_2" : (1963,1968),    
# "hindcast_fold_1" : (1958,1962),     
   }

# CPC = True
CPC = False
# CPC_John = True

varname = 'T2m'
# varname = 'H500'
if CPC:
    varnameVerif = 'tavg'
else:
    varnameVerif = 'T2m_NorthAmerica'


select_month = False
months_set = [None]

# select_month = True
# months_set = [(11, 4), (5, 10)]

offsets = [False]
offsets = [True]
offsets = [False,True]
# persistence = True
persistence = False

# ocn = True
ocn = False

events = ['cold','warm']
for offset in offsets:
    for event in events:
        for months in months_set:
            anom_list = []
            jra55_list = []
            for forecast_mode, period in forecast_periods_input.items():
                print(f'forecast_mode : {forecast_mode}, {period}')
                print(f'offset = {offset}, CPC = {CPC}, select_month = {select_month}, months = {months}, event = {event}')
                years= np.arange(period[0],period[1]+1,1).tolist() # These are years in each fold.
                # year_list.append(period[0])
                # year_list.append(period[1])
        
                if forecast_mode == 'hindcast_fold_1':
                    start_date = f'{period[0]}-01-08'
                else:
                    start_date = f'{period[0]}-01-01'
                
                end_date   = f'{period[1]}-12-31'
                date_range = pd.date_range(start=start_date, end=end_date)
                anomvar = varname+'_anom'
                if persistence:
                    VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_persistence'
                    files = [os.path.join(VERIFDIR,varname,f'{varname}.{year}.persistence.week34.nc') for year in years]
                    # print(files)
                elif ocn:
                    VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_optimal_climate_normals'
                    # file_ocn = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_optimal_climate_normals/T2m/T2m.1968-2022.ocn.week34.nc'
                    files = [os.path.join(VERIFDIR,varname,f'{varname}.1968-2022.ocn.week34.nc')]

                else:            
                    VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_{expt_name}_{forecast_mode}'

                    if offset:
                        files = [os.path.join(VERIFDIR, date,            varname,f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
                    else:
                        files = [os.path.join(VERIFDIR, date,'no_offset',varname,f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
                ds = xr.open_mfdataset(files)
                ds = check_lat_order(ds,verbose=False)

                ##### Read in forecasts #####
                if ocn:
                    print('in ocn')
                    if forecast_mode == 'hindcast_fold_10' or forecast_mode == 'reforecast':
                        # The ocn file is wk34 average already and contains 1968-2022 in one file.
                        print(f'reading ocn file {files}')
                        anom = ds[varname]
                else:
                    for label,lt in zip(['wk34'],[(21,28)]):
                        # new dataset with current lead time. if more than one, concatenate lead times
                        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
                        anom   = newds[anomvar]

                ##### Read in verification #####
                if CPC:
                    jrafiles = [f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/CPC_verification/{varnameVerif}/{varnameVerif}.{year}.week34.nc' for year in years]
                    jra55 = xr.open_mfdataset(jrafiles,combine='nested', concat_dim='time')

                else:
                    if offset:    
                        jrafiles = [f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{year}.week34.add_offset.nc' for year in years]
                        jra55 = xr.open_mfdataset(jrafiles,combine='nested', concat_dim='time')
                        varnameVerif = 'T2m_wk34_NorthAmerica'
                    else:
                        jrafiles = [f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{year}.week34.nc' for year in years]
                        jra55 = xr.open_mfdataset(jrafiles,combine='nested', concat_dim='time')
                        varnameVerif = 'T2m_NorthAmerica'
                    # jrafiles = [f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{year}.week34.nc' for year in years]
                    # jra55 = xr.open_mfdataset(jrafiles,combine='nested', concat_dim='time')
                jra55 = check_lat_order(jra55,verbose=False)
                jra55 = jra55[varnameVerif].sel(time=slice(start_date,end_date))
                
                if ocn:
                    if forecast_mode == 'hindcast_fold_10' or forecast_mode == 'reforecast':
                        anom_list.append(anom)
                else:            
                    anom_list.append(anom)
                    
                jra55_list.append(jra55)
            anom  = xr.concat(anom_list ,dim='time').sortby('time')
            jra55 = xr.concat(jra55_list,dim='time').sortby('time')
            allyears = set(anom.time.dt.year.values)

            print('!!!!! anom and jra55:!!!!!')
            print('anom:',anom)    
            print('jra55:',jra55)    

            # Uncomment these lines if using IFS dates
            # file = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/IFS/T2m/T2m.1997-2016.week34.nc'
            file = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/IFS/T2m/T2m.2017-2022.week34.nc'
            dsIFS = xr.open_dataset(file)
            timeIFS = dsIFS.time
            allyears = set(timeIFS.dt.year.values)

            anom  = anom.sel(time=timeIFS)
            jra55 = jra55.sel(time=timeIFS)
            
            print('!!!!! after selecting IFS period:!!!!!')
            print('anom:',anom)    
            print('jra55:',jra55)   
            
            if select_month:
                if months == (11,4):            
                    vCPC = jra55.sel(time=(jra55['time.month'] >= months[0]) | (jra55['time.month'] <= months[1]))
                    ANOM =  anom.sel(time=( anom['time.month'] >= months[0]) | ( anom['time.month'] <= months[1]))
                elif months == (5,10):
                    vCPC = jra55.sel(time=(jra55['time.month'] >= months[0]) & (jra55['time.month'] <= months[1]))
                    ANOM =  anom.sel(time=( anom['time.month'] >= months[0]) &  (anom['time.month'] <= months[1]))
                # print(vCPC.time.dt.month[0::30])
                # print(ANOM.time.dt.month[0::30])
                # print(ANOM)
            else:
                vCPC = jra55
                ANOM = anom
            HSS = heidke_skill_score_map_cold_warm(vCPC, ANOM, event)
                
            # HSS = heidke_skill_score_map(vCPC, ANOM)
            dsHSS = xr.DataArray(HSS,
                coords={'lat': ('lat', ds.lat.values), 'lon': ('lon', ds.lon.values)}, dims=['lat', 'lon']
                )
            dsHSS = dsHSS.to_dataset(name=f'HSS_{event}')

            os.system(f'mkdir -p {VERIFDIR}/verification')
            if persistence or ocn:
                if select_month:
                    fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.{event}.map.nc'
                    # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.{event}.map.nc'
                else: 
                    fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.{event}.map.nc'
                    # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.week34.{event}.map.nc'
                print(f'persistence = {persistence}, ocn = {ocn},fout = {fout}')
            else:    
                if offset:
                    if CPC:
                        if select_month:
                            fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.add_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.{event}.map.nc'
                        else:
                            fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.add_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.week34.{event}.map.nc'

                    else:
                        if select_month:
                            fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.add_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.add_offset.{event}.map.nc'
                        else:
                            fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.add_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.week34.add_offset.{event}.map.nc'
                else:
                    if CPC:
                        if select_month:
                            fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                        else:
                            fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.no_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.week34.no_offset.{event}.map.nc'
                    else:
                        if select_month:
                            fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                        else:
                            fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.no_offset.{event}.map.nc'
                            # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.week34.no_offset.{event}.map.nc'
                print(f'offset = {offset}, fout = {fout}')
            
            os.system(f'rm -f {fout}')
            dsHSS.to_netcdf(fout)
