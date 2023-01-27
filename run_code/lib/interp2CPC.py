#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:38:00 2022

@author: slillo

Edited: J.R. Albers 10.4.2022

"""


###################################################################################
# IMPORT PACKAGES
####################################################################################

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt,timedelta
import copy
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
warnings.simplefilter("ignore")

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib import tools
from lib.tools import *
from lib import plot
from lib.plot import PlotMap
from lib import driver
# from LIM_CPC.tools import *
# from LIM_CPC.plot import PlotMap
# from LIM_CPC import driver


####################################################################################
# MAIN CODE BODY
####################################################################################

LIMdriver = driver.Driver('namelist.py')
LIMdriver.get_variables(read=True)
LIMdriver.get_eofs(read=True)

varobj = LIMdriver.use_vars['T2m']['data']
limlon = varobj.lon
limlat = varobj.lat

cpcmask = xr.open_dataset('data_clim/cpcmask.nc')
l1,l2 = np.meshgrid(cpcmask.lon.data,cpcmask.lat.data)
a = copy.copy(cpcmask.mask1.data)
cpclon = l1[np.where(~np.isnan(a))].flatten()
cpclat = l2[np.where(~np.isnan(a))].flatten()

latwt = np.cos(np.radians(cpclat))
pthresh = 55

def interp2CPC(lat,lon,z):

    xMin = max([0,min(lon)-5])
    yMin = max([-90,min(lat)-5])
    xMax = min([360,max(lon)+5])
    yMax = min([90,max(lat)+5])

    # grid the data.
    zLL = z[np.argmin((lon-xMin)**2+(lat-yMin)**2)]
    zLR = z[np.argmin((lon-xMax)**2+(lat-yMin)**2)]
    zUL = z[np.argmin((lon-xMin)**2+(lat-yMax)**2)]
    zUR = z[np.argmin((lon-xMax)**2+(lat-yMax)**2)]
    lonNew = np.array(list(lon)+[xMin,xMax,xMin,xMax])
    latNew = np.array(list(lat)+[yMin,yMin,yMax,yMax])
    zNew = np.array(list(z)+[zLL,zLR,zUL,zUR])

    zmap = griddata((lonNew,latNew), zNew, (cpcmask.lon.data[None,:], cpcmask.lat.data[:,None]), method='cubic')

    out = zmap[np.where(~np.isnan(cpcmask.mask1.data))]

    return out

year = dt.now().year
ds1 = xr.open_mfdataset([f'data_realtime/cpctmin.{year+i}.nc' for i in range(-5,1,1)])
ds2 = xr.open_mfdataset([f'data_realtime/cpctmax.{year+i}.nc' for i in range(-5,1,1)])
ds3 = ds1.merge(ds2,compat='override')
ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
obs = ds4.drop(('tmin','tmax'))

clim = xr.open_dataset('data_clim/tavg.day.1991-2020.ltm.nc')
clim = xr.concat([clim,clim],dim='time')
obsroll = obs.rolling({'time':7},min_periods=int(7//2)).mean()
climroll = clim.drop_vars(['climatology_bounds']).rolling({'time':7},min_periods=int(7//2)).mean()
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

print('loaded stuff')

#%%

def fillnan(a):
    amask = np.isnan(a)
    a[amask] = np.interp(np.flatnonzero(amask), np.flatnonzero(~amask), a[~amask])
    return a
def fillzero(a):
    amask = (a==0)
    a[amask] = np.interp(np.flatnonzero(amask), np.flatnonzero(~amask), a[~amask])
    return a

def make_verif_maps(T_INIT):

    # try:
    #     VERIFDIR = f'../Images/{T_INIT:%Y%m%d}'
    #     ds = xr.open_dataset(f'{VERIFDIR}/T2m.{T_INIT:%Y%m%d}.nc')

    if T_INIT<dt(2021,5,29):
        VERIFDIR = f'../Images/{T_INIT:%Y%m%d}_blend'
        ds = xr.open_dataset(f'{VERIFDIR}/T2m.{T_INIT:%Y%m%d}.nc')
    else:
        VERIFDIR = f'../Images/{T_INIT:%Y%m%d}'
        ds = xr.open_dataset(f'{VERIFDIR}/T2m_offset.{T_INIT:%Y%m%d}.nc')

    skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

    for label,lt in zip(['wk2','wk3','wk4','wk34'],[(14,),(21,),(28,),(21,28)]):

        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
   
        if len(newds.T2m_anom.data.shape)>2:
            anom = varobj.flatten(newds.T2m_anom.data[-1])
        else:
            anom = varobj.flatten(newds.T2m_anom.data[:,:])    
        anom = fillnan(anom)
        ANOM = interp2CPC(limlat,limlon,anom)

        if len(newds.T2m_anom.data.shape)>2:
            spread = varobj.flatten(newds.T2m_spread.data[-1])
        else:
            spread = varobj.flatten(newds.T2m_spread.data[:])
        spread = fillnan(spread)
        spread = fillzero(spread)
        SPREAD = interp2CPC(limlat,limlon,spread)

        a = np.mean([tmp.sel(time=T_INIT+timedelta(days=l)).tavg.data for l in lt],axis=0)
        a = a*cpcmask.mask1.data[1:-1,1:-1]
        whereNan = np.argwhere(~np.isnan(a))
        ilat,ilon = [*zip(*whereNan)]
        z = a[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1]
        lat = latbins[min(ilat):max(ilat)+2]
        lon = lonbins[min(ilon):max(ilon)+2]

        vCPC = np.array([i for i in z.flatten() if not np.isnan(i)])
        bounds = [-np.inf*np.ones(len(vCPC)),np.zeros(len(vCPC)),np.inf*np.ones(len(vCPC))]
        OBS = get_categorical_obs((vCPC,),bounds)[0]

        bounds = [-np.inf*np.ones(len(ANOM)),np.zeros(len(ANOM)),np.inf*np.ones(len(ANOM))]
        PROB = get_categorical_fcst((ANOM,),(SPREAD,),bounds)[0]

        validwhere = (abs(PROB[1]-.5)<(pthresh/100-.5))
        HSS = get_heidke(PROB.T,OBS.T,weights=latwt,categorical=True)
        HSS_thresh = get_heidke(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=True)
        RPSS = get_rpss(PROB.T,OBS.T,weights=latwt,categorical=False)
        RPSS_thresh = get_rpss(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=False)
        if np.any(validwhere)==False:
            RPSS_thresh = np.nan
            HSS_thresh = np.nan

        if lt == (21,28):
            skill_dict = {'date':T_INIT,'HSS':HSS,'HSS_55':HSS_thresh,'RPSS':RPSS,'RPSS_55':RPSS_thresh}
        else:
            skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

        hitmiss = ((PROB[1]-.5)*(OBS[1]-.5)>0).astype(int)*2-1
        hitmiss[validwhere] = 0
        hMAP = copy.copy(cpcmask.mask1.data)
        hMAP[np.where(~np.isnan(hMAP))] = hitmiss
        hMAP = hMAP[min(ilat)+1:max(ilat)+2,min(ilon)+1:max(ilon)+2]

        colors = ['lightpink','w','mediumseagreen']
        cmap = mcolors.ListedColormap(colors)

        fig = plt.figure(figsize=(12,8))
        m = PlotMap(projection='dynamic',lon=lon,lat=lat,res='m',\
                    central_longitude=-90)
        m.setup_ax()
        ax=m.ax
        p = m.pcolor(lon,lat,hMAP,cmap=cmap,vmin=-1,vmax=1)
        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        m.drawstates(linewidth=0.5,color='0.25',zorder=10)
        m.ax.axis([-170+90,-50+90,15,75])
        m.ax.set_title(f'T2m hit / miss  >{pthresh}%',loc='left',fontweight='bold',fontsize=14)
        m.ax.set_title(f'Init: {T_INIT:%a %d %b %Y}'+f'\nValid: {T_INIT+timedelta(days=min(lt)-6):%d %b} – {T_INIT+timedelta(days=max(lt)):%d %b %Y}',
                     loc='right',fontsize=14)
        ax.text( 0.03, 0.24, f'Heidke (all) = {HSS:.3f} \nHeidke (>{pthresh}%) = {HSS_thresh:.3f}'+\
                f'\nRPSS (all) = {RPSS:.3f} \nRPSS (>{pthresh}%) = {RPSS_thresh:.3f}',
                ha='left', va='center', transform=ax.transAxes,fontsize=12,zorder=99)
        cbar = plt.colorbar(p,ticks=[-.67,.67],orientation='horizontal',ax=m.ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
        cbar.ax.set_xticklabels(['Miss','Hit'])

        ltlab = '-'.join([f'{l:03}' for l in lt])
        plt.savefig(f'{VERIFDIR}/T2m_lt{ltlab}_hitmiss_55.png',bbox_inches='tight',dpi=150)

        plt.close()

        # for destination in copy_to_dirs:
        #     os.system(f'cp -r {VERIFDIR}/* {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
        #     os.system(f'rm {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}/*.nc')

    return skill_dict



def make_verif_maps_CPCperiod(T_INIT,dayoffset):

    # try:
    #     VERIFDIR = f'../Images/{T_INIT:%Y%m%d}'
    #     ds = xr.open_dataset(f'{VERIFDIR}/T2m.{T_INIT:%Y%m%d}.nc')

    if T_INIT<dt(2021,5,29):
        VERIFDIR = f'../Images/{T_INIT:%Y%m%d}_blend'
        ds = xr.open_dataset(f'{VERIFDIR}/T2m.{T_INIT:%Y%m%d}.nc')
    else:
        VERIFDIR = f'../Images/{T_INIT:%Y%m%d}'
        ds = xr.open_dataset(f'{VERIFDIR}/T2m_Week_34_official_CPC_period.{T_INIT:%Y%m%d}.nc')

    skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

    for label,lt in zip(['wk34'],[(21+dayoffset,28+dayoffset)]):

        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
   
        if len(newds.T2m_anom.data.shape)>2:
            anom = varobj.flatten(newds.T2m_anom.data[-1])
        else:
            anom = varobj.flatten(newds.T2m_anom.data[:,:])    
        anom = fillnan(anom)
        ANOM = interp2CPC(limlat,limlon,anom)

        if len(newds.T2m_anom.data.shape)>2:
            spread = varobj.flatten(newds.T2m_spread.data[-1])
        else:
            spread = varobj.flatten(newds.T2m_spread.data[:])
        spread = fillnan(spread)
        spread = fillzero(spread)
        SPREAD = interp2CPC(limlat,limlon,spread)

        a = np.mean([tmp.sel(time=T_INIT+timedelta(days=l)).tavg.data for l in lt],axis=0)
        a = a*cpcmask.mask1.data[1:-1,1:-1]
        whereNan = np.argwhere(~np.isnan(a))
        ilat,ilon = [*zip(*whereNan)]
        z = a[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1]
        lat = latbins[min(ilat):max(ilat)+2]
        lon = lonbins[min(ilon):max(ilon)+2]

        vCPC = np.array([i for i in z.flatten() if not np.isnan(i)])
        bounds = [-np.inf*np.ones(len(vCPC)),np.zeros(len(vCPC)),np.inf*np.ones(len(vCPC))]
        OBS = get_categorical_obs((vCPC,),bounds)[0]

        bounds = [-np.inf*np.ones(len(ANOM)),np.zeros(len(ANOM)),np.inf*np.ones(len(ANOM))]
        PROB = get_categorical_fcst((ANOM,),(SPREAD,),bounds)[0]

        validwhere = (abs(PROB[1]-.5)<(pthresh/100-.5))
        HSS = get_heidke(PROB.T,OBS.T,weights=latwt,categorical=True)
        HSS_thresh = get_heidke(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=True)
        RPSS = get_rpss(PROB.T,OBS.T,weights=latwt,categorical=False)
        RPSS_thresh = get_rpss(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=False)
        if np.any(validwhere)==False:
            RPSS_thresh = np.nan
            HSS_thresh = np.nan

        if lt == (21,28):
            skill_dict = {'date':T_INIT,'HSS':HSS,'HSS_55':HSS_thresh,'RPSS':RPSS,'RPSS_55':RPSS_thresh}
        else:
            skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

        hitmiss = ((PROB[1]-.5)*(OBS[1]-.5)>0).astype(int)*2-1
        hitmiss[validwhere] = 0
        hMAP = copy.copy(cpcmask.mask1.data)
        hMAP[np.where(~np.isnan(hMAP))] = hitmiss
        hMAP = hMAP[min(ilat)+1:max(ilat)+2,min(ilon)+1:max(ilon)+2]

        colors = ['lightpink','w','mediumseagreen']
        cmap = mcolors.ListedColormap(colors)

        fig = plt.figure(figsize=(12,8))
        m = PlotMap(projection='dynamic',lon=lon,lat=lat,res='m',\
                    central_longitude=-90)
        m.setup_ax()
        ax=m.ax
        p = m.pcolor(lon,lat,hMAP,cmap=cmap,vmin=-1,vmax=1)
        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        m.drawstates(linewidth=0.5,color='0.25',zorder=10)
        m.ax.axis([-170+90,-50+90,15,75])
        m.ax.set_title(f'T2m hit / miss  >{pthresh}%',loc='left',fontweight='bold',fontsize=14)
        m.ax.set_title(f'Init: {T_INIT:%a %d %b %Y}'+f'\nValid: {T_INIT+timedelta(days=min(lt)-6):%d %b} – {T_INIT+timedelta(days=max(lt)):%d %b %Y}',
                     loc='right',fontsize=14)
        ax.text( 0.03, 0.24, f'Heidke (all) = {HSS:.3f} \nHeidke (>{pthresh}%) = {HSS_thresh:.3f}'+\
                f'\nRPSS (all) = {RPSS:.3f} \nRPSS (>{pthresh}%) = {RPSS_thresh:.3f}',
                ha='left', va='center', transform=ax.transAxes,fontsize=12,zorder=99)
        cbar = plt.colorbar(p,ticks=[-.67,.67],orientation='horizontal',ax=m.ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
        cbar.ax.set_xticklabels(['Miss','Hit'])

        ltlab = '-'.join([f'{l:03}' for l in lt])
        plt.savefig(f'{VERIFDIR}/T2m_CPC_period_lt{ltlab}_hitmiss_55.png',bbox_inches='tight',dpi=150)

        plt.close()

        # for destination in copy_to_dirs:
        #     os.system(f'cp -r {VERIFDIR}/* {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}')
        #     os.system(f'rm {destination}{T_INIT:%Y%m}/{T_INIT:%Y%m%d}/*.nc')

    return skill_dict

#%%


# T_INIT = dt(2021,5,28)

# while T_INIT>dt(2017,1,1):

#     print(T_INIT)

#     cpus = mp.cpu_count()

#     DAYS = [T_INIT-timedelta(i) for i in range(cpus)]

#     with mp.Pool(cpus) as pool:
#         pool.map(make_verif_maps,DAYS)

#     T_INIT = T_INIT-timedelta(cpus)
