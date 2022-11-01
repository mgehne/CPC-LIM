r"""

Edited: J.R. Albers 10.4.2022

Description of what this code actually does needs to be added...(it is for sure used though)

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime as dt,timedelta
import os
import multiprocessing as mp
import warnings

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib import plot
from lib.plot import PlotMap
# from LIM_CPC.plot import PlotMap


####################################################################################
# MAIN CODE BODY
####################################################################################

warnings.simplefilter("ignore")

#%%
# cpcmask = xr.open_dataset('data_clim/cpcmask.nc')

# ds1 = xr.open_mfdataset([f'data_realtime/cpctmin.{2022+i}.nc' for i in range(-5,1,1)])
# ds2 = xr.open_mfdataset([f'data_realtime/cpctmax.{2022+i}.nc' for i in range(-5,1,1)])
# ds3 = ds1.merge(ds2,compat='override')
# ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
# obs = ds4.drop(('tmin','tmax'))

# clim = xr.open_dataset('data_clim/tavg.day.1981-2010.ltm.nc')
# #clim = xr.open_dataset('data_clim/tavg.day.1991-2020.ltm.nc')
# clim = xr.concat([clim,clim],dim='time')
# obsroll = obs.rolling({'time':7},min_periods=int(7//2)).mean()
# climroll = clim.drop_vars(['climatology_bounds']).rolling({'time':7},min_periods=int(7//2)).mean()
# obsroll = obsroll.assign_coords({'dayofyear':('time',[(t.dayofyear-1)%365+1 for t in pd.DatetimeIndex(obsroll.time.data)])})

# obsgroup = obsroll.groupby("dayofyear")
# climatology = climroll.groupby("time.dayofyear").mean("time")
# anom = obsgroup-climatology

# latbins = np.mean([cpcmask.lat.data[:-1],cpcmask.lat.data[1:]],axis=0)
# lonbins = np.mean([cpcmask.lon.data[:-1],cpcmask.lon.data[1:]],axis=0)

# tmp = anom.groupby_bins('lat',latbins).mean()
# tmp = tmp.groupby_bins('lon',lonbins).mean()
# tmp['lat'] = cpcmask.lat.data[1:-1]
# tmp['lon'] = cpcmask.lon.data[1:-1]
# tmp['tavg'] = tmp['tavg'].T

# print('loaded stuff')

#%%
def getCPCobs(dates,per=1,savetopath=None):

    if not isinstance(dates,(tuple,list)):
        dates = [dates]

    #Get CPC global temp data for most recent year (this year or last)
    #For tmax and tmin
    year = max(dates).year
    if year==dt.now().year:
        try:
            os.system(f'wget -O data_realtime/cpctmax.{year}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{year}.nc')
            os.system(f'wget -O data_realtime/cpctmin.{year}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year}.nc')
            yr0 = year
        except:
            os.system(f'wget -O data_realtime/cpctmax.{year-1}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{year-1}.nc')
            os.system(f'wget -O data_realtime/cpctmin.{year-1}.nc https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year-1}.nc')
            yr0 = year-1
    else:
        yr0 = year

    ds1 = xr.open_mfdataset([f'data_realtime/cpctmin.{yr0-1}.nc',f'data_realtime/cpctmin.{yr0}.nc'])
    ds2 = xr.open_mfdataset([f'data_realtime/cpctmax.{yr0-1}.nc',f'data_realtime/cpctmax.{yr0}.nc'])
    ds3 = ds1.merge(ds2,compat='override')
    ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
    obs = ds4.drop(('tmin','tmax'))

    clim = xr.open_dataset('data_clim/tavg.day.1991-2020.ltm.nc')
    clim = xr.concat([clim,clim],dim='time')
    cpcmask = xr.open_dataset('data_clim/cpcmask.nc')

    if per>1:
        obsroll = obs.rolling({'time':per},min_periods=int(per//2)).mean()
        climroll = clim.drop_vars(['climatology_bounds']).rolling({'time':per},min_periods=int(per//2)).mean()
    else:
        obsroll = obs
        climroll = clim.drop_vars(['climatology_bounds'])

    obsroll = obsroll.assign_coords({'dayofyear':('time',[(t.dayofyear-1)%365+1 for t in pd.DatetimeIndex(obsroll.time.data)])})

    obsgroup = obsroll.groupby("dayofyear")
    climatology = climroll.groupby("time.dayofyear").mean("time")
    anom = obsgroup-climatology

    latbins = np.mean([cpcmask.lat.data[:-1],cpcmask.lat.data[1:]],axis=0)
    lonbins = np.mean([cpcmask.lon.data[:-1],cpcmask.lon.data[1:]],axis=0)

    tmp = anom.groupby_bins('lat',latbins).mean()
    tmp = tmp.groupby_bins('lon',lonbins).mean()
    tmp['lat'] = cpcmask.lat.data[1:-1]
    tmp['lon'] = cpcmask.lon.data[1:-1]
    tmp['tavg'] = tmp['tavg'].T

    for date in dates:
        #print(date,per)

        a = tmp.sel(time=date).tavg.data
        a = a*cpcmask.mask1.data[1:-1,1:-1]
        whereNan = np.argwhere(~np.isnan(a))
        ilat,ilon = [*zip(*whereNan)]
        z = a[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1]
        lat = latbins[min(ilat):max(ilat)+2]
        lon = lonbins[min(ilon):max(ilon)+2]

        colors1 = ['#8200dc','#1e3cff','#00a0ff','#00c8c8','0.9']
        colors2 = ['0.9','#e6dc32','#e6af2d','#f08228','#fa3c3c']
        levels1 = [np.arange(l,[-4,-2,-1.5,-1,-.5,0][i+1],.01) for i,l in enumerate([-4,-2,-1.5,-1,-.5,0][:-1])]
        levels2 = [np.arange(l,[0,.5,1,1.5,2,4][i+1],.01) for i,l in enumerate([0,.5,1,1.5,2,4][:-1])]
        colors = [i for l,c in zip(levels1,colors1) for i in [c]*len(l)]+\
                [i for l,c in zip(levels2,colors2) for i in [c]*len(l)]
        cmap = mcolors.ListedColormap(colors)
        cmap.set_over('#8b0000')
        cmap.set_under('#a000c8')

        fig = plt.figure(figsize=(12,8))
        m = PlotMap(projection='dynamic',lon=lon,lat=lat,res='m',\
                    central_longitude=-90)
        m.setup_ax()
        p = m.pcolor(lon,lat,z,cmap=cmap,vmin=-4,vmax=4)
        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        m.drawstates(linewidth=0.5,color='0.25',zorder=10)
        m.ax.axis([-170+90,-50+90,15,75])
        m.ax.set_title('T2m anomaly',loc='left',fontweight='bold',fontsize=14)
        m.ax.set_title(f'Valid: {date-timedelta(days=per-1):%d %b} – {date:%d %b %Y}',
                     loc='right',fontsize=14)
        cbar = plt.colorbar(p,ticks=[-4,-2,-1.5,-1,-.5,.5,1,1.5,2,4],orientation='horizontal',extend='both',ax=m.ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
        cbar.ax.set_xticklabels([-4,-2,-1.5,-1,-.5,.5,1,1.5,2,4])
        cbar_label = '$^oC$'
        cbar.ax.set_xlabel(cbar_label,fontsize=14)
        if savetopath is not None:
            plt.savefig(f'{savetopath}/CPCverif_anom.{date:%Y%m%d}.{per:03}.png',bbox_inches='tight',dpi=150)
            plt.close()

        colors = ['#77b5e2','#df723e']
        cmap = mcolors.ListedColormap(colors)

        fig = plt.figure(figsize=(12,8))
        m = PlotMap(projection='dynamic',lon=lon,lat=lat,res='m',\
                    central_longitude=-90)
        m.setup_ax()
        p = m.pcolor(lon,lat,np.where(np.isnan(z),np.nan,(z>0)),cmap=cmap,vmin=0,vmax=1)
        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        m.drawstates(linewidth=0.5,color='0.25',zorder=10)
        m.ax.axis([-170+90,-50+90,15,75])
        m.ax.set_title('T2m category',loc='left',fontweight='bold',fontsize=14)
        m.ax.set_title(f'Valid: {date-timedelta(days=per-1):%d %b} – {date:%d %b %Y}',
                     loc='right',fontsize=14)
        cbar = plt.colorbar(p,ticks=[.25,.75],orientation='horizontal',ax=m.ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
        cbar.ax.set_xticklabels(['Cold','Warm'])
        if savetopath is not None:
            plt.savefig(f'{savetopath}/CPCverif_cat.{date:%Y%m%d}.{per:03}.png',bbox_inches='tight',dpi=150)
            plt.close()


def make_CPC_maps(T_INIT):
    copy_to_dirs = ['/httpd-test/psd/forecasts/lim_s2s/']
    LIMpage_path = '../Images'

    try:

        if T_INIT<dt(2021,5,29):
            VERIFDIR = f'../Images/{T_INIT:%Y%m%d}_blend'
            ds = xr.open_dataset(f'{VERIFDIR}/T2m.{T_INIT:%Y%m%d}.nc')

        getCPCobs([T_INIT+timedelta(days=i) for i in (7,14,21,28)],per=7,savetopath=VERIFDIR)
        getCPCobs(T_INIT+timedelta(days=28),per=14,savetopath=VERIFDIR)

        for destination in copy_to_dirs:
            os.system(f'rm -r {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
            os.system(f'cp -r {VERIFDIR} {destination}{T_INIT:%Y%m}')
            os.system(f'mv {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}_blend {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
            os.system(f'rm {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}/*.nc')

    except:
        pass


#%%

# T_INIT = dt(2021,5,28)

# while T_INIT>dt(2017,1,1):

#     print(T_INIT)

#     cpus = mp.cpu_count()

#     DAYS = [T_INIT-timedelta(i) for i in range(cpus)]

#     with mp.Pool(cpus) as pool:
#         pool.map(make_CPC_maps,DAYS)

#     T_INIT = T_INIT-timedelta(cpus)
