#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:39:58 2021

@author: slillo
"""


import numpy as np
import os
import xarray as xr

os.system('wget -P data_clim https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.day.1991-2020.ltm.nc')
os.system('wget -P data_clim https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.day.1991-2020.ltm.nc')

ds1 = xr.open_dataset('data_clim/tmin.day.1991-2020.ltm.nc')
ds2 = xr.open_dataset('data_clim/tmax.day.1991-2020.ltm.nc')
ds3 = ds1.merge(ds2,compat='override')
ds4 = ds3.assign(tavg=(("time","lat","lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
ds4.drop(('tmin','tmax','valid_yr_count')).to_netcdf('data_clim/tavg.day.1991-2020.ltm.nc')

#%%

import numpy as np
import os
import xarray as xr

for year in range(1979,1991):

    os.system(f'wget -O data_clim/cpcdata/tmax.{year}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{year}.nc')
    os.system(f'wget -O data_clim/cpcdata/tmin.{year}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year}.nc')

    ds1 = xr.open_dataset(f'data_clim/cpcdata/tmin.{year}.nc')
    ds2 = xr.open_dataset(f'data_clim/cpcdata/tmax.{year}.nc')
    ds3 = ds1.merge(ds2,compat='override')
    ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
    obs = ds4.drop(('tmin','tmax'))
    
    cpcmask = xr.open_dataset('data_clim/cpcmask.nc')
            
    latbins = np.mean([cpcmask.lat.data[:-1],cpcmask.lat.data[1:]],axis=0)
    lonbins = np.mean([cpcmask.lon.data[:-1],cpcmask.lon.data[1:]],axis=0)
            
    tmp = obs.groupby_bins('lat',latbins).mean().groupby_bins('lon',lonbins).mean()
    tmp['lat'] = cpcmask.lat.data[1:-1]
    tmp['lon'] = cpcmask.lon.data[1:-1]
    tmp['temp'] = (['time','lat','lon'],tmp.tavg.data.T)
    tmp = tmp.drop(('lat_bins','lon_bins','tavg'))
    
    tmp['temp'] = tmp['temp']*cpcmask.mask1.data[1:-1,1:-1]
    whereNan = np.argwhere(~np.isnan(tmp.temp.data[-1]))
    ilat,ilon = [*zip(*whereNan)]
    tmp = tmp.isel(lat=slice(min(ilat),max(ilat)+1),lon=slice(min(ilon),max(ilon)+1))
    
    tmp.to_netcdf(f'data_clim/cpcdata/tavg.{year}.nc')


#%%

import pandas as pd
import xarray as xr
from LIM_CPC import driver
LIMdriver = driver.Driver('namelist.py')
LIMdriver.get_eofs(read=True)

ds = xr.open_dataset('data_clim/tmp_CPC/EOF_T2m.nc')

for month in range(1,13):

    eofobj = LIMdriver.eofobjs[month]['T2m']
    time = eofobj.varobjs[0].time
    pc = eofobj.pc
    JRA = {pd.to_datetime(t):p for t,p in zip(time,pc)}
    CPC = {pd.to_datetime(t):p for t,p in zip(ds.time.data,ds.pc.data) if pd.to_datetime(t) in JRA.keys()}
    JRA = {t:p for t,p in JRA.items() if t in CPC.keys()}
    
    JRACPC = np.array([[JRA[t][:5],CPC[t][:5]] for t in JRA.keys()]).swapaxes(0,1)
    
    C0 = np.matmul(JRACPC[0].T, JRACPC[0]) / (JRACPC[0].shape[0] - 1) 
    Ctau = np.matmul(JRACPC[1].T, JRACPC[0]) / (JRACPC[1].shape[0] - 1) 
        
    G = np.matmul(Ctau, np.linalg.pinv(C0))

#%%

date = dt(1995,1,15)

pcsJ = pc[list(time).index(date)][:5]
CPCout = np.matmul(G, np.matrix(pcsJ).T)

recon = np.sum(np.array(CPCout).squeeze()[:,None,None]*ds.eof_T2m[:5],axis=0)
out = recon*varobj.climo_stdev/np.sqrt(np.cos(np.radians(ds.lat_T2m.data))[:,None])

eofobj.varobjs[0].plot_map(time=date)
plt.show()
plt.pcolor(out,vmin=-12,vmax=12,cmap='bwr')



#%%


from scipy.interpolate import NearestNDInterpolator
import xarray as xr
import numpy as np

ds = xr.open_dataset('data_clim/tavg.day.1991-2020.ltm.nc')
data = ds.tavg.data

new_data = []
for iday,dataday in enumerate(data):
    print(iday)
    mask = np.where(~np.isnan(dataday))
    interp = NearestNDInterpolator(np.transpose(mask), dataday[mask])
    filled_data = interp(*np.indices(dataday.shape))
    
    new_data.append(filled_data)

#%%

from LIM_CPC import driver
from LIM_CPC.tools import *

LIMdriver = driver.Driver('namelist.py')
LIMdriver.get_variables(read=True)

varobj = LIMdriver.use_vars['T2m']['data']

NEWCLIM_INTERP = np.array([interp2LIM(ds.lat.data,ds.lon.data,var_day+273.15,varobj) for var_day in new_data])

climo = gfilt(3*list(NEWCLIM_INTERP),[15]+[0]*len(NEWCLIM_INTERP.shape[1:]))[365:2*365]

coords = {"time": {'dims':('time',),
                     'data':ds.time.data,
                     'attrs':{'long_name':'initial time',}},
        "lon": {'dims':("pts",),
                  'data':varobj.lon,
                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
        "lat": {'dims':("pts",),
                  'data':varobj.lat,
                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
        }

vardict = {"T2m": {'dims':("time","pts"),
                               'data':climo,}
                }

save_ncds(vardict,coords,filename='data_clim/CPC.1991-2020.nc')