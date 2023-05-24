#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:13:51 2021

@author: slillo

Edited: J.R. Albers 10.4.2022
This function is used to create retrospective (out-of-sample) reforecasts using the NOAA PSL/CPC subseasonal LIM.

- Forecasts are saved as netCDF files via the directories LIMpage_path and FCSTDIR
- LIM forecast operator pickles must already have been created; if they haven't, then read=False must be inserted into the LIMdriver.get_variables() and LIMdriver.get_eofs() calls
- As currently set up, the forecasts use the add_offset flag, which adjusts the reference climatology of the anomalies to be that of the current NOAA CPC base period (currently 1991-2020)

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
import copy
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from calendar import monthrange
# import multiprocessing as mp
import pickle
import traceback

# Edited import method J.R. ALbers 10.4.2022
# import lib
# from lib import data_retrieval
# from lib import dataset
from lib import driver
# from lib import plot 
from lib.tools import get_categorical_obs, get_categorical_fcst, get_heidke, get_rpss
# from lib import verif
from lib.tools import *
# from LIM_CPC import driver
# import data_retrieval
# import LIM_CPC
# from LIM_CPC.tools import *

from lib import getCPCobs
from lib.getCPCobs import *
#from lib import interp2CPC
#from lib.interp2CPC import make_verif_maps, make_verif_maps_CPCperiod

import warnings
warnings.filterwarnings('ignore')


####################################################################################
### BEGIN USER INPUT ###

#LIMpage_path = f'../Images'
LIMpage_path = f'../Images_retrospective_python_with_soil_moisture_new_data_retrospective_lon'
# LIMpage_path = f'/home/ycheng/LIM/skill_check'
FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_regression' 
PLOTDIR = f'{FCSTDIR}/Images_regression'
SKILLDIR = f'{LIMpage_path}/skill_pickles_regression'
#FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_beta'
#PLOTDIR = f'{FCSTDIR}/Images_adjClim'
#SKILLDIR = f'{LIMpage_path}/skill_pickles_adjClim'
# RETROdata_path = './data_retrospective' # CYM, not used, commented out

# varname = 'CPCtemp'
varname = 'T2m'

T_START = dt(2018,1,1) #dt(YEAR,MONTH,1)
# T_START = dt(2018,3,18) #dt(YEAR,MONTH,1)
T_END = dt(2022,12,31) #dt(YEAR,MONTH,LASTDAY)
VERIFDAYS = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

### END USER INPUT ###
####################################################################################



####################################################################################
# verification function
####################################################################################

# LIMdriver = driver.Driver('namelist_retrospective.py')
LIMdriver = driver.Driver(f'namelist_retrospective_beta.py')
LIMdriver.get_variables(read=True)
#LIMdriver.get_eofs(read=True)

varobj = LIMdriver.use_vars[varname]['data']
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

def fillnan(a):
    amask = np.isnan(a)
    a[amask] = np.interp(np.flatnonzero(amask), np.flatnonzero(~amask), a[~amask])
    return a

def fillzero(a):
    amask = (a==0)
    a[amask] = np.interp(np.flatnonzero(amask), np.flatnonzero(~amask), a[~amask])
    return a    

# copmute observed CPC anomalies from climatology - read in last 5 years for now
now = dt.now()
endyear = np.min([now.year+1,int(T_END.year)+2])
years =  np.arange( int(T_START.year),endyear,1)

ds1 = xr.open_mfdataset([f'data_retrospective/cpctmin.{year}.nc' for year in years])
ds2 = xr.open_mfdataset([f'data_retrospective/cpctmax.{year}.nc' for year in years])
ds3 = ds1.merge(ds2,compat='override')
ds4 = ds3.assign(tavg=(("time", "lat", "lon"),np.nanmean([ds3.tmin,ds3.tmax],axis=0)))
obs = ds4.drop(('tmin','tmax'))
obsroll = obs.rolling({'time':7},min_periods=int(7//2)).mean()
obsroll = obsroll.assign_coords({'dayofyear':('time',[(t.dayofyear-1)%365+1 for t in pd.DatetimeIndex(obsroll.time.data)])})
obsgroup = obsroll.groupby("dayofyear")

clim = xr.open_dataset('data_clim/tavg.day.1991-2020.ltm.nc')
clim = xr.concat([clim,clim],dim='time')
climroll = clim.drop_vars(['climatology_bounds']).rolling({'time':7},min_periods=int(7//2)).mean()
climatology = climroll.groupby("time.dayofyear").mean("time")
climP = xr.open_dataset('data_clim/tavg.day.1981-2010.ltm.nc')
climP = xr.concat([climP,climP],dim='time')
climProll = climP.drop_vars(['climatology_bounds']).rolling({'time':7},min_periods=int(7//2)).mean()
climatologyP = climProll.groupby("time.dayofyear").mean("time")

anom = obsgroup-climatology
anomP = obsgroup-climatologyP

latbins = np.mean([cpcmask.lat.data[:-1],cpcmask.lat.data[1:]],axis=0)
lonbins = np.mean([cpcmask.lon.data[:-1],cpcmask.lon.data[1:]],axis=0)

tmpC = anom.groupby_bins('lat',latbins).mean()
tmpC = tmpC.groupby_bins('lon',lonbins).mean()
tmpC['lat'] = cpcmask.lat.data[1:-1]
tmpC['lon'] = cpcmask.lon.data[1:-1]
tmpC['tavg'] = tmpC['tavg'].T

tmpP = anomP.groupby_bins('lat',latbins).mean()
tmpP = tmpP.groupby_bins('lon',lonbins).mean()
tmpP['lat'] = cpcmask.lat.data[1:-1]
tmpP['lon'] = cpcmask.lon.data[1:-1]
tmpP['tavg'] = tmpP['tavg'].T


def make_verif_maps(T_INIT,varname,VERIFDIR,climoffset=False):
    """
    Compute scores and plot verification maps for given inital date.
    """

    # open model forecast netcdf file for current intial date
    if climoffset:
        ds = xr.open_dataset(f'{VERIFDIR}/{varname}_offset.{T_INIT:%Y%m%d}.nc')
    else:    
        ds = xr.open_dataset(f'{VERIFDIR}/{varname}.{T_INIT:%Y%m%d}.nc')
    anomvar = varname+'_anom'
    # anomvar = varname
    spreadvar = varname+'_spread'

    # initialize skill dictionary
    skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

    # CPC observed anomalies from 1981-2010 climo (tmpP) or 1991-2020 climo (tmpC)
    if T_INIT<dt(2021,5,29):
        tmp = tmpP
    else:
        tmp = tmpC    

    # loop through lead times
    for label,lt in zip(['wk3','wk4','wk34'],[(21,),(28,),(21,28)]):
    # for label,lt in zip(['wk0','wk1','wk2','wk3','wk4','wk34'],[(0,),(7,),(14,),(21,),(28,),(21,28)]):
    # for label,lt in zip(['wk1','wk2','wk3','wk4','wk34'],[(7,),(14,),(21,),(28,),(21,28)]):

        # new dataset with current lead time. if more than one, concatenate lead times
        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
   
        # flatten forecast anomaly and bin to 2degree CPC grid
        if len(newds[anomvar].data.shape)>2:
            anom = varobj.flatten(newds[anomvar].data[-1])
        else:
            anom = varobj.flatten(newds[anomvar].data[:,:])    
        anom = fillnan(anom)
        ANOM = interp2CPC(limlat,limlon,anom)
        
        if len(newds[anomvar].data.shape)>2:
            spread = varobj.flatten(newds[spreadvar].data[-1])
        else:
            spread = varobj.flatten(newds[spreadvar].data[:])
        spread = fillnan(spread)
        spread = fillzero(spread)
        SPREAD = interp2CPC(limlat,limlon,spread)

        # read obs for the valid times of the forecast
        a = np.mean([tmp.sel(time=T_INIT+timedelta(days=l)).tavg.data for l in lt],axis=0)
        a = a*cpcmask.mask1.data[1:-1,1:-1]
        whereNan = np.argwhere(~np.isnan(a))
        ilat,ilon = [*zip(*whereNan)]
        z = a[min(ilat):max(ilat)+1,min(ilon):max(ilon)+1]
        lat = latbins[min(ilat):max(ilat)+2]
        lon = lonbins[min(ilon):max(ilon)+2]

        # compute categorial obs
        vCPC = np.array([i for i in z.flatten() if not np.isnan(i)])
        bounds = [-np.inf*np.ones(len(vCPC)),np.zeros(len(vCPC)),np.inf*np.ones(len(vCPC))]
        OBS = get_categorical_obs((vCPC,),bounds)[0]
        # compute categorical forecast
        bounds = [-np.inf*np.ones(len(ANOM)),np.zeros(len(ANOM)),np.inf*np.ones(len(ANOM))]
        PROB = get_categorical_fcst((ANOM,),(SPREAD,),bounds)[0]

        # compute skill scores
        validwhere = (abs(PROB[1]-.5)<(pthresh/100-.5))
        HSS = get_heidke(PROB.T,OBS.T,weights=latwt,categorical=True)
        HSS_thresh = get_heidke(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=True)
        RPSS = get_rpss(PROB.T,OBS.T,weights=latwt,categorical=False)
        RPSS_thresh = get_rpss(PROB.T[~validwhere],OBS.T[~validwhere],weights=latwt[~validwhere],categorical=False)
        if np.any(validwhere)==False:
            RPSS_thresh = np.nan
            HSS_thresh = np.nan

        # add skill scores to dictionary for return values
        if lt == (21,28):
        # if lt == (0,7,14,21,28):
        # if lt == (7,14,21,28):
            skill_dict = {'date':T_INIT,'HSS':HSS,'HSS_55':HSS_thresh,'RPSS':RPSS,'RPSS_55':RPSS_thresh}
        else:
            skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

        # plot HitMiss map
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
        if climoffset:
            plt.savefig(f'{PLOTDIR}/HitMissMaps/{varname}_offset_lt{ltlab}_hitmiss_55_{T_INIT:%Y%m%d}.png',bbox_inches='tight',dpi=150)
        else:    
            plt.savefig(f'{PLOTDIR}/HitMissMaps/{varname}_lt{ltlab}_hitmiss_55_{T_INIT:%Y%m%d}.png',bbox_inches='tight',dpi=150)

        plt.close()

    return skill_dict

def make_verif_maps_CPCperiod(T_INIT,varname,VERIFDIR,weekday,dayoffset,climoffset=False):

    if climoffset:
        ds = xr.open_dataset(f'{VERIFDIR}/{varname}_Week_34_official_CPC_period_climo_offset_weekday{weekday}.{T_INIT:%Y%m%d}.nc')
    else: 
        ds = xr.open_dataset(f'{VERIFDIR}/{varname}_Week_34_official_CPC_period_weekday{weekday}.{T_INIT:%Y%m%d}.nc')
    anomvar = varname+'_anom'
    # anomvar = varname
    spreadvar = varname+'_spread'

    skill_dict = {'date':T_INIT,'HSS':np.nan,'HSS_55':np.nan,'RPSS':np.nan,'RPSS_55':np.nan}

    # CPC observed anomalies from 1981-2010 climo (tmpP) or 1991-2020 climo (tmpC)
    if T_INIT<dt(2021,5,29):
        tmp = tmpP
    else:
        tmp = tmpC  

    for label,lt in zip(['wk34'],[(21+dayoffset,28+dayoffset)]):

        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
   
        if len(newds[anomvar].data.shape)>2:
            anom = varobj.flatten(newds[anomvar].data[-1])
        else:
            anom = varobj.flatten(newds[anomvar].data[:,:])    
        anom = fillnan(anom)
        ANOM = interp2CPC(limlat,limlon,anom)
        
        if len(newds[anomvar].data.shape)>2:
            spread = varobj.flatten(newds[spreadvar].data[-1])
        else:
            spread = varobj.flatten(newds[spreadvar].data[:])
        spread = fillnan(spread)
        spread = fillzero(spread)
        SPREAD = interp2CPC(limlat,limlon,spread)

        a = np.mean([tmp.sel(time=T_INIT+timedelta(days=l)).tavg for l in lt],axis=0)
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

        if lt == (21+dayoffset,28+dayoffset):
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
        if climoffset:
            plt.savefig(f'{PLOTDIR}/HitMissMaps/{varname}_offset_CPC_period_weekday{str(weekday)}_lt{ltlab}_hitmiss_55_{T_INIT:%Y%m%d}.png',bbox_inches='tight',dpi=150)
        else: 
            plt.savefig(f'{PLOTDIR}/HitMissMaps/{varname}_CPC_period_weekday{str(weekday)}_lt{ltlab}_hitmiss_55_{T_INIT:%Y%m%d}.png',bbox_inches='tight',dpi=150)

        plt.close()

    return skill_dict


####################################################################################
# Verification for BLENDED LIM
####################################################################################

for T_INIT_verif in VERIFDAYS:

    weekday = T_INIT_verif.weekday()
    dayoffset = (4-weekday)%7

    try:
        VERIFDIR = f'{FCSTDIR}'

        print(f'DOING VERIFICATION FOR {T_INIT_verif:%Y%m%d}')
        print('make verification maps and skill scores')

        if os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.CPCperiod.TuesdayInit.p') or \
            os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.CPCperiod.FridayInit.p') or \
                (weekday in (0,2,3,5,6) and os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.p')):
            print(f'Skill file {SKILLDIR}/{T_INIT_verif:%Y%m%d}.p exists.')
        else:
            # get CPC obs for the valid forecast times
            getCPCobs([T_INIT_verif+timedelta(days=i) for i in (21,28)],per=7,savetopath=f'{PLOTDIR}/VerifMaps')
            # getCPCobs(T_INIT_verif+timedelta(days=28),per=14,savetopath=f'{PLOTDIR}/VerifMaps')
            # getCPCobs([T_INIT_verif+timedelta(days=i) for i in (0,7,14,21,28)],per=7,savetopath=f'{PLOTDIR}/VerifMaps')
            getCPCobs(T_INIT_verif+timedelta(days=28),per=14,savetopath=f'{PLOTDIR}/VerifMaps')
            #try:
            skill = make_verif_maps(T_INIT_verif,varname,VERIFDIR)
            pickle.dump(skill, open( f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.p','wb'))
            ds = xr.Dataset(skill)
            ds.to_netcdf(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.nc')
            ds.close()
            #except:    
            #    pass
            try:
                if weekday==1 or weekday==4:
                    # get CPC obs for the valid forecast times
                    getCPCobs([T_INIT_verif+timedelta(days=i) for i in (21+dayoffset,28+dayoffset)],per=7,savetopath=f'{VERIFDIR}/Images_adjClim')
                    # getCPCobs(T_INIT_verif+timedelta(days=28+dayoffset),per=14,savetopath=f'{PLOTDIR}/VerifMaps')# CYM leave it as it is
                    skill = make_verif_maps_CPCperiod(T_INIT_verif,varname,VERIFDIR,weekday,dayoffset)
        
                    if weekday==1:
                        CPCname = 'CPCperiod.TuesdayInit'
                    elif weekday==4:
                        CPCname = 'CPCperiod.FridayInit'   
                    pickle.dump(skill, open( f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.{CPCname}.p','wb'))
                    ds = xr.Dataset(skill)
                    ds.to_netcdf(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.{CPCname}.nc')
                    ds.close()
            except:
                pass

        # if os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.CPCperiod.TuesdayInit.p') or \
        #     os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.CPCperiod.FridayInit.p') or \
        #         (weekday in (0,2,3,5,6) and os.path.isfile(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.p')):
        #     print(f'Skill file {SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.p exists.')
        # else:
        #     skill = make_verif_maps(T_INIT_verif,varname,VERIFDIR,climoffset=True)
        #     pickle.dump(skill, open( f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.p','wb'))
        #     ds = xr.Dataset(skill)
        #     ds.to_netcdf(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.offset.nc')
        #     ds.close()
        #     try:
        #         if weekday==1 or weekday==4:
        #             skill = make_verif_maps_CPCperiod(T_INIT_verif,varname,VERIFDIR,weekday,dayoffset,climoffset=True)
        
        #             if weekday==1:
        #                 CPCname = 'offset.CPCperiod.TuesdayInit'
        #             elif weekday==4:
        #                 CPCname = 'offset.CPCperiod.FridayInit'   
        #             pickle.dump(skill, open( f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.{CPCname}.p','wb'))
        #             ds = xr.Dataset(skill)
        #             ds.to_netcdf(f'{SKILLDIR}/{T_INIT_verif:%Y%m%d}.{CPCname}.nc')
        #             ds.close()
        #     except:
        #         pass    

        # MAKE SKILL PLOTS
        dates = [T_INIT_verif+timedelta(days=i) for i in range(-364,1,1)]

        skill_dict = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}
        skill_dict_Friday = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}
        skill_dict_Tuesday = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}

        # skill_dict_offset = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}
        # skill_dict_Friday_offset = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}
        # skill_dict_Tuesday_offset = {'date':[],'HSS':[],'HSS_55':[],'RPSS':[],'RPSS_55':[]}

        for T_INIT in dates:
            weekday = T_INIT.weekday()
            dayoffset = (4-weekday)%7
            try:
                skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.p', 'rb' ) )
                for k,v in skill.items():
                    skill_dict[k].append(v)
                # skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.offset.p', 'rb' ) )
                # for k,v in skill.items():
                #     skill_dict_offset[k].append(v)    
                if dayoffset==0:
                    skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.CPCperiod.FridayInit.p', 'rb' ) )
                    for k,v in skill.items():
                        skill_dict_Friday[k].append(v) 
                    # skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.offset.CPCperiod.FridayInit.p', 'rb' ) )
                    # for k,v in skill.items():
                        # skill_dict_Friday_offset[k].append(v)    
                if dayoffset==3:             
                    skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.CPCperiod.TuesdayInit.p', 'rb' ) )
                    for k,v in skill.items():
                        skill_dict_Tuesday[k].append(v) 
                    # skill = pickle.load( open( f'{SKILLDIR}/{T_INIT:%Y%m%d}.offset.CPCperiod.TuesdayInit.p', 'rb' ) )
                    # for k,v in skill.items():
                        # skill_dict_Tuesday_offset[k].append(v)                      
            except:
                pass      
        print(skill_dict['HSS'])
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
        plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_HSS_timeseries_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        plt.close()

        # HSS with climo offset
        # fig = plt.figure(figsize=(10,6),dpi=200)
        # time = skill_dict_offset['date']
        # HSS = skill_dict_offset['HSS']
        # HSS_55 = skill_dict_offset['HSS_55']

        # HSS_avg = f'{np.nanmean(HSS):0.3f}'
        # HSS_55_avg = f'{np.nanmean(HSS_55):0.3f}'

        # plt.plot(time,HSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{HSS_avg: >16}')
        # plt.plot(time,HSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{HSS_55_avg: >10}')

        # plt.yticks(np.arange(-1,1.1,.2))
        # xlim = plt.gca().get_xlim()
        # plt.plot(xlim,[0,0],'k',linewidth=1.5)
        # plt.axis([*xlim,-1.1,1.1])
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

        # plt.title('Temperature Week-3/4 Heidke Skill Score w/ climo offset',fontsize=17)
        # plt.xlabel('Initialization Time',fontsize=15)
        # plt.ylabel('HSS',fontsize=15)
        # plt.legend(loc='lower left',fontsize=10.5)
        # plt.grid()
        # plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_offset_HSS_timeseries_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        # plt.close()

        #RPSS
        fig = plt.figure(figsize=(10,6),dpi=150)
        time = skill_dict['date']
        RPSS = skill_dict['RPSS']
        RPSS_55 = skill_dict['RPSS_55']
        #RPSS_CPC = skill_dict_CPC['RPSS']
        #RPSS_55_CPC = skill_dict_CPC['RPSS_55']

        RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
        RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'
        #RPSS_CPC_avg = f'{np.nanmean(RPSS_CPC):0.3f}'
        #RPSS_55_CPC_avg = f'{np.nanmean(RPSS_55_CPC):0.3f}'

        plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{RPSS_avg: >16}')
        plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{RPSS_55_avg: >10}')
        #plt.plot(time_CPC,RPSS_CPC,color='dodgerblue',linestyle='dashed',label=f'{"CPC period CONUS": <12}'+f'{RPSS_CPC_avg: >16}')
        #plt.plot(time_CPC,RPSS_55_CPC,color='darkorange',linestyle='dashed',label=f'{"CPC period CONUS >55%": <12}'+f'{RPSS_55_CPC_avg: >10}')

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
        plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_RPSS_timeseries_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        plt.close()

        # RPSS with climo offset
        # fig = plt.figure(figsize=(10,6),dpi=150)
        # time = skill_dict_offset['date']
        # RPSS = skill_dict_offset['RPSS']
        # RPSS_55 = skill_dict_offset['RPSS_55']

        # RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
        # RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'

        # plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS": <12}'+f'{RPSS_avg: >16}')
        # plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS >55%": <12}'+f'{RPSS_55_avg: >10}')

        # plt.yticks(np.arange(-1,1.1,.2))
        # xlim = plt.gca().get_xlim()
        # plt.plot(xlim,[0,0],'k',linewidth=1.5)
        # plt.axis([*xlim,-1.1,1.1])
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

        # plt.title('Temperature Week-3/4 Ranked Probability Skill Score w/ climo offset',fontsize=17)
        # plt.xlabel('Initialization Time',fontsize=15)
        # plt.ylabel('RPSS',fontsize=15)
        # plt.legend(loc='lower left',fontsize=10.5)
        # plt.grid()
        # plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_offset_RPSS_timeseries_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        # plt.close()

        #HSS - Tuesday/Friday forecast
        fig = plt.figure(figsize=(10,6),dpi=200)
        time = skill_dict_Friday['date']
        HSS = skill_dict_Friday['HSS']
        HSS_55 = skill_dict_Friday['HSS_55']
        time_CPC = skill_dict_Tuesday['date']
        time_CPC = [date + timedelta(days=3) for date in time_CPC]
        HSS_CPC = skill_dict_Tuesday['HSS']
        HSS_55_CPC = skill_dict_Tuesday['HSS_55']

        HSS_avg = f'{np.nanmean(HSS):0.3f}'
        HSS_55_avg = f'{np.nanmean(HSS_55):0.3f}'
        HSS_CPC_avg = f'{np.nanmean(HSS_CPC):0.3f}'
        HSS_55_CPC_avg = f'{np.nanmean(HSS_55_CPC):0.3f}'

        plt.plot(time,HSS,color='dodgerblue',label=f'{"CONUS (Friday Init)": <12}'+f'{HSS_avg: >16}')
        plt.plot(time,HSS_55,color='darkorange',label=f'{"CONUS (Friday Init) >55%": <12}'+f'{HSS_55_avg: >10}')
        plt.plot(time_CPC,HSS_CPC,color='dodgerblue',linestyle='dashed',label=f'{"CPC period CONUS (Tuesday Init)": <12}'+f'{HSS_CPC_avg: >16}')
        plt.plot(time_CPC,HSS_55_CPC,color='darkorange',linestyle='dashed',label=f'{"CPC period CONUS (Tuesday Init) >55%": <12}'+f'{HSS_55_CPC_avg: >10}')

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
        plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_HSS_timeseries_CPCperiod_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        plt.close()


        #RPSS - Tuesday/Friday forecast
        fig = plt.figure(figsize=(10,6),dpi=150)
        RPSS = skill_dict['RPSS']
        RPSS_55 = skill_dict['RPSS_55']
        time = skill_dict_Friday['date']
        RPSS = skill_dict_Friday['RPSS']
        RPSS_55 = skill_dict_Friday['RPSS_55']
        time_CPC = skill_dict_Tuesday['date']
        time_CPC = [date + timedelta(days=3) for date in time_CPC]
        RPSS_CPC = skill_dict_Tuesday['RPSS']
        RPSS_55_CPC = skill_dict_Tuesday['RPSS_55']

        RPSS_avg = f'{np.nanmean(RPSS):0.3f}'
        RPSS_55_avg = f'{np.nanmean(RPSS_55):0.3f}'
        RPSS_CPC_avg = f'{np.nanmean(RPSS_CPC):0.3f}'
        RPSS_55_CPC_avg = f'{np.nanmean(RPSS_55_CPC):0.3f}'

        plt.plot(time,RPSS,color='dodgerblue',label=f'{"CONUS (Friday Init)": <12}'+f'{RPSS_avg: >16}')
        plt.plot(time,RPSS_55,color='darkorange',label=f'{"CONUS (Friday Init) >55%": <12}'+f'{RPSS_55_avg: >10}')
        plt.plot(time_CPC,RPSS_CPC,color='dodgerblue',linestyle='dashed',label=f'{"CPC period CONUS (Tuesday Init)": <12}'+f'{RPSS_CPC_avg: >16}')
        plt.plot(time_CPC,RPSS_55_CPC,color='darkorange',linestyle='dashed',label=f'{"CPC period CONUS (Tuesday Init) >55%": <12}'+f'{RPSS_55_CPC_avg: >10}')

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
        plt.savefig(f'{PLOTDIR}/TimeSeries/{varname}_RPSS_timeseries_CPCperiod_{T_INIT_verif:%Y%m%d}.png',bbox_inches='tight')
        plt.close()

        #for destination in copy_to_dirs:
        #    os.system(f'rm -r {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}')
        #    os.system(f'cp -r {VERIFDIR} {destination}{T_INIT_verif:%Y%m}')
        #    os.system(f'rm {destination}{T_INIT_verif:%Y%m}/{T_INIT_verif:%Y%m%d}/*.nc')

    except:
        print(traceback.format_exc())
        print(f'couldnt make verif for {T_INIT_verif}')













# # INITIALIZE AND RUN LIM FORECAST
# print('\nInitializing and running LIM...')
# #LIMdriver = driver.Driver(f'namelist_{YEAR}{MONTH:02}.py')
# LIMdriver = driver.Driver(f'namelist_retrospective.py')
# LIMdriver.get_variables()
# LIMdriver.get_eofs()
# LIMdriver.prep_realtime_data(limkey=1)

# for T_INIT in hindcastdays:
#     START = dt.now()
#     try:
#         LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,28),fullVariance=True)
#         if T_INIT<dt(2021,5,29):
#             climoffsetfile = 'data_clim/CPC.1981-2010.nc'
#         else:
#             climoffsetfile = 'data_clim/CPC.1991-2020.nc'    
#         LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(21,28),save_to_path=FCSTDIR,add_offset=climoffsetfile)
#     except:
#         print(f'{T_INIT:%Y%m%d} data is unavailable and/or forecast was unable to run')
#         pass

#     FINISH = dt.now()
#     ELAPSED = (FINISH-START).total_seconds()/60
#     print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
