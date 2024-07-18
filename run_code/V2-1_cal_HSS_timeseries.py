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
warnings.filterwarnings('ignore')
import scipy.io


expt_name="fixed_58-16_climo"
# expt_name="v2p0"

forecast_periods_input = {
# "reforecast"     :  (2019,2022),
"reforecast"     :  (2017,2022),
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
varname = 'T2m'
# varname = 'H500'

add_offsets = [False,True]
# add_offsets = [False]
# add_offsets = [True]

# persistence = True
persistence = False

# IFS = True
IFS = False

USonly = True
# USonly = False


for add_offset in add_offsets:
    for forecast_mode, period in forecast_periods_input.items():
        print(f'forecast_mode : {forecast_mode}, {period}')
        years= np.arange(period[0],period[1]+1,1).tolist()

        for year in years:
            print(year)
            if year ==1958:
                start_date = f'{year}-01-08'
            else:
                start_date = f'{year}-01-01'
            
            end_date   = f'{year}-12-31'
            date_range = pd.date_range(start=start_date, end=end_date)
            print(f'add_offset = {add_offset}, persistence = {persistence}, USonly = {USonly}')
            anomvar = varname+'_anom'
            if persistence:
                VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_persistence'
                files = [os.path.join(VERIFDIR,varname,f'{varname}.{year}.persistence.week34.nc')]
                print(files)
            elif IFS:
                 # file = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/IFS/T2m/T2m.2017-2022.week34.nc'
                # dsIFS = xr.open_dataset(file)
                # timeIFS = dsIFS.time
                VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/IFS'
                files = [os.path.join(VERIFDIR,varname,f'T2m.2017-2022.week34.nc')]
                print(files)
            else:            
                VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_{expt_name}_{forecast_mode}'

                if add_offset:
                    files = [os.path.join(VERIFDIR, date,            varname,f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
                else:
                    files = [os.path.join(VERIFDIR, date,'no_offset',varname,f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
            ds = xr.open_mfdataset(files)
            ds = check_lat_order(ds,verbose=False)
            if IFS:
                ds = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
                print(f'IFS time = {ds.time}')
            dates = ds.time
            # labels = ['wk3','wk4','wk34']
            # lts    = [(21,),(28,), (21,28)]
            labels = ['wk34']
            lts    = [(21,28)]
            HSS={}
            # for label,lt in zip(['wk34'],[(21,28)]):
            for label,lt in zip(labels,lts):
                # new dataset with current lead time. if more than one, concatenate lead times
                print(f'calculating {label}, day {lt}')
                if IFS:
                    anom = ds[varname]# this wouldn't work for other lead for IFS
                else:
                    newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
                    anom   = newds[anomvar]
                if add_offset:
                    fname=f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{year}.week34.add_offset.nc'
                    jra55 = xr.open_dataset(fname)
                    print(fname)
                else:
                    jra55 = xr.open_dataset(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{year}.week34.nc')
                    print('jra55 no offset')
                # jra55 = xr.open_dataset(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/{varname}/{varname}.{date_range[0].year}.week34.all.nc')
                jra55 = check_lat_order(jra55,verbose=False)

                # This is only because verification for no_offset has not been updated to have _wk34 suffix yet.
                if add_offset: 
                    if USonly:
                        # jra55 = jra55[varname].sel(time=slice(start_date,end_date))# need to make wk3, wk4 change here
                        jra55 = jra55[f'{varname}_{label}'].sel(time=dates)# need to make wk3, wk4 change here
                    else: 
                        jra55 = jra55[f'{varname}_NorthAmerica_{label}'].sel(time=dates)
                        # print(jra55)
                else:
                    if USonly:
                        # jra55 = jra55[varname].sel(time=slice(start_date,end_date))# need to make wk3, wk4 change here
                        jra55 = jra55[f'{varname}'].sel(time=dates)# need to make wk3, wk4 change here
                    else: 
                        jra55 = jra55[f'{varname}_NorthAmerica'].sel(time=dates)
                        # print(jra55)

                jra55 = jra55.sel(time=dates) # subset the verification to only dates with a forecast (for IFS)
                if jra55.time.shape != anom.time.shape:
                    print('validation and forecasts are off differnt time size')
                    break
                
                skill_HSS = []
                # for i, date in enumerate(date_range):
                for i, date in enumerate(dates):
                    if i%50 == 0:
                        print(date)
                    vCPC = jra55.isel(time=i)
                    ANOM = anom.isel(time=i)
                    if i==0:
                        # dsmask = xr.open_dataset('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/v2p0_verification/mask.nc')
                        # dsmask = check_lat_order(dsmask)
                        # maskUS = dsmask['mask']
                        # mask   = xr.where(ANOM.isnull() | vCPC.isnull() | maskUS.isnull() ,np.nan  ,1.)
                        # This mask is included in (jra55) verification already!
                        mask   = xr.where(ANOM.isnull() | vCPC.isnull() ,np.nan  ,1.)
                        
                        l1,l2 = np.meshgrid(mask.lon.data,mask.lat.data)
                        LATS = xr.DataArray(l2,dims=("lat", "lon"),
                        coords={
                            "lat": ("lat", mask.lat.data),
                            "lon": ("lon", mask.lon.data)})
                        latwt = np.cos(np.radians(LATS))**.5

                    HSStmp = heidke_skill_score(vCPC, ANOM, mask, latwt)
                    skill_HSS.append(HSStmp)
                HSS[f'HSS_{label}'] = skill_HSS

            # dsHSS= xr.Dataset({"HSS":(['time'],skill_HSS)}, coords={'time':date_range})
            dsHSS= xr.Dataset({var:(['time'],data) for var, data in HSS.items()}, coords={'time':dates})
            
            os.system(f'mkdir -p {VERIFDIR}/verification')
            if persistence:
                if USonly:
                    fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{date_range[0].year}.week34.add_offset.US.nc'
                print(f'persistence, fout = {fout}')
            elif IFS:
                if USonly:
                    if add_offset:
                        fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{year}.week34.add_offset.US.nc'
                    else:
                        fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{year}.week34.no_offset.US.nc'
                print(f'IFS, fout = {fout}')
            else:    
                if add_offset:
                    if USonly:
                        fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{year}.week34.add_offset.US.nc'
                else:
                    if USonly:
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{date_range[0].year}.week34.no_offset.US.nc'
                        fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{year}.week34.no_offset.US.nc'
                    else: 
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{date_range[0].year}.week34.no_offset.NorthAmerica.nc'
                        fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{year}.week34.all.no_offset.NorthAmerica.nc'
                print(f'add_offset = {add_offset}, fout = {fout}')
            
            os.system(f'rm -f {fout}')
            dsHSS.to_netcdf(fout)
