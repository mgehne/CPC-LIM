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

def cal_ACC_spatial_map(dsX,dsY,time_centering=True):
    """
    Calculate the anomaly correlation for two xarray dataset
    Parameters:
    dsX(time,lat,lon): The first xr.dataset
    dsY(time,lat,lon): The second xr.dataset
    
    Returns:
    An xarray of ACC(lat,lon)
    """
    if time_centering:
        print('removing time mean')
        xAnom = dsX - dsX.mean(dim='time')
        yAnom = dsY - dsY.mean(dim='time')
    else:
        xAnom = dsX
        yAnom = dsY

    xyCov = (xAnom*yAnom).sum(dim=('time'))
    xAnom2 = (xAnom**2).sum(dim='time')
    yAnom2 = (yAnom**2).sum(dim='time')
    ACC = xyCov/(xAnom2*yAnom2)**0.5
    dsACC = xr.DataArray(ACC,dims=['lat','lon'],coords={'lat':dsX.lat,'lon':dsX.lon})
    return dsACC

expt_name="fixed_58-16_climo"
# expt_name="v2p0"

forecast_periods_input = {
# # "reforecast"     :  (2017,2022),
"hindcast_fold_10": (2011,2016),
"hindcast_fold_9" : (2005,2010),    
"hindcast_fold_8" : (1999,2004),    
"hindcast_fold_7" : (1993,1998),    
"hindcast_fold_6" : (1987,1992),    
"hindcast_fold_5" : (1981,1986),    
"hindcast_fold_4" : (1975,1980),    
"hindcast_fold_3" : (1969,1974),    
"hindcast_fold_2" : (1963,1968),    
"hindcast_fold_1" : (1958,1962),     
}



varname = 'T2m'
# varname = 'H500'
varnameVerif = 'T2m_NorthAmerica'


select_month = False
months_set = [None]

# select_month = True
# months_set = [(11, 4), (5, 10)]

offsets = [False]
# offsets = [True]
# offsets = [False,True]

year_list = [1958,2016]
events = ['cold','warm']
for offset in offsets:
    for event in events:
        for months in months_set:
            diri ='/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/hindcasts_data/'
            
            anom = xr.open_dataset(f'{diri}{expt_name}_lim_1958_2022.nc')
            jra55 = xr.open_dataset(f'{diri}{expt_name}_jra_1958_2022.nc')
            
            anom = anom.sel(time=slice(year_list[0],year_list[1]))
            jra55 = jra55.sel(time=slice(year_list[0],year_list[1]))
            
            if select_month:
                if months == (11,4):            
                    vCPC = jra55.sel(time=(jra55['time.month'] >= months[0]) | (jra55['time.month'] <= months[1]))
                    ANOM =  anom.sel(time=( anom['time.month'] >= months[0]) | ( anom['time.month'] <= months[1]))
                elif months == (5,10):
                    vCPC = jra55.sel(time=(jra55['time.month'] >= months[0]) & (jra55['time.month'] <= months[1]))
                    ANOM =  anom.sel(time=( anom['time.month'] >= months[0]) &  (anom['time.month'] <= months[1]))
            else:
                vCPC = jra55
                ANOM = anom
            print('-----------------------')
            print('After select_month')
            print(vCPC.time.dt.year[0::30], vCPC.time.dt.month[0::30])
            print(ANOM.time.dt.year[0::30], ANOM.time.dt.month[0::30])
            print(ANOM)
            print('-----------------------')
            
            if event  == 'warm':
                vCPC_event = xr.where(jra55 >= 0., jra55, np.nan)
            elif event == 'cold':
                vCPC_event = xr.where(jra55  < 0., jra55, np.nan)
            
            ANOM_event = xr.where(~vCPC_event.isnull() , ANOM , np.nan)
            
            HSS = heidke_skill_score_map_cold_warm(vCPC, ANOM, event)
                
            # HSS = heidke_skill_score_map(vCPC, ANOM)
            dsHSS = xr.DataArray(HSS,
                coords={'lat': ('lat', ds.lat.values), 'lon': ('lon', ds.lon.values)}, dims=['lat', 'lon']
                )
            dsHSS = dsHSS.to_dataset(name=f'HSS_{event}')

            os.system(f'mkdir -p {VERIFDIR}/verification')
    
            if offset:
                if not select_month:
                    fout = f'{VERIFDIR}/verification/ACC.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.week34.{event}.map.nc'
                else:
                    if select_month:
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.add_offset.{event}.map.nc'
                        fout = f'{VERIFDIR}/verification/ACC.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.add_offset.{event}.map.nc'
                    else:
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.add_offset.{event}.map.nc'
                        fout = f'{VERIFDIR}/verification/ACC.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.week34.add_offset.{event}.map.nc'
            else:
                if not select_month:
                    fout = f'{VERIFDIR}/verification/ACC.against.CPC.{varname}.{min(allyears)}-{max(allyears)}.week34.no_offset.{event}.map.nc'                    
                else:
                    if select_month:
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                        fout = f'{VERIFDIR}/verification/ACC.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.{months[0]}-{months[1]}.week34.no_offset.{event}.map.nc'
                    else:
                        # fout = f'{VERIFDIR}/verification/HSS.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.ifs.period.week34.no_offset.{event}.map.nc'
                        fout = f'{VERIFDIR}/verification/ACC.against.JRA.{varname}.{min(allyears)}-{max(allyears)}.week34.no_offset.{event}.map.nc'
            print(f'offset = {offset}, fout = {fout}')
        
            os.system(f'rm -f {fout}')
            dsHSS.to_netcdf(fout)
