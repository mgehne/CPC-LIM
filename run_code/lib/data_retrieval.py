#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:58:51 2020

@author: slillo
"""

# =============================================================================
# Python Script to retrieve XX online Data files of 'ds628.8',
# total 3.76M. This script uses 'requests' to download data.
#
# Highlight this script by Select All, Copy and Paste it into a file;
# make the file executable and run it on command line.
#
# You need pass in your password as a parameter to execute
# this script; or you can set an environment variable RDAPSWD
# if your Operating System supports it.
#
# Contact davestep@ucar.edu (Dave Stepaniak) for further assistance.
# =============================================================================

# =============================================================================
# Variables to grab:
#
# time = datetime object 
#
# Isobaric heights, 2.5 deg: f'anl_p25/{time:%Y%m}/anl_p25_hgt.{time:%Y%m%d%H}'
# Isobaric streamfunction, 2.5 deg: f'anl_p25/{time:%Y%m}/anl_p25_strm.{time:%Y%m%d%H}'
# Sea level pressure, 1.25 deg: f'anl_surf125/{time:%Y%m}/anl_surf125.{time:%Y%m%d%H}'
# 2m Temperature, 1.25 deg: "
# Soil moisture, 1.25 deg: f'anl_land125/{time:%Y%m}/anl_surf125.{time:%Y%m%d%H}'
# colIrr:
# (+) Down SW rad flux at nominal top 
# (-) Down SW rad flux at surface
# (-) Down LW rad flux at surface
# (-) Up SW rad flux at nominal top
# (+) Up SW rad flux at surface
# (-) Up LW rad flux at nominal top
# (+) Up LW rad flux at surface
# (+) Up sensible heat flux at surface
# (+L*) Precipitation
# 
# f'fcst_phy2m125/{time-timedelta(days=1):%Y%m}/fcst_phy2m125.{time-timedelta(days=1):%Y%m%d%H}'
#
# =============================================================================

import sys, os
import requests
from datetime import datetime as dt,timedelta
import cfgrib
import xarray as xr

class getData:
    
    def __init__(self,email,password,savetopath):
        self.email = email
        self.password = password
        self.savetopath = savetopath
                
    def download(self,days):
        
        self.days = [d.replace(hour=0,minute=0,second=0,microsecond=0) for d in days]
                
        def check_file_status(filepath, filesize):
            sys.stdout.write('\r')
            sys.stdout.flush()
            size = int(os.stat(filepath).st_size)
            percent_complete = (size/filesize)*100
            sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
            sys.stdout.flush()
        
        url = 'https://rda.ucar.edu/cgi-bin/login'
        values = {'email' : self.email, 'passwd' : self.password, 'action' : 'login'}
        # Authenticate
        ret = requests.post(url,data=values)
        if ret.status_code != 200:
            print('Bad Authentication')
            print(ret.text)
            exit(1)
        dspath = 'https://rda.ucar.edu/data/ds628.8/'
        
        daytimes3 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,3)]
        daytimes6 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,6)]
        
        self.filedict = {\
        'hgt':[f'anl_p25/{t:%Y%m}/anl_p25_hgt.{t:%Y%m%d%H}' for t in daytimes6],\
        'surf':[f'anl_surf125/{t:%Y%m}/anl_surf125.{t:%Y%m%d%H}' for t in daytimes6],\
        'land':[f'anl_land125/{t:%Y%m}/anl_land125.{t:%Y%m%d%H}' for t in daytimes6],\
        'phy2m':[f'fcst_phy2m125/{t:%Y%m}/fcst_phy2m125.{t:%Y%m%d%H}' for t in daytimes3]}
        
        #filelist = [i for j in self.filedict.values() for i in j]
        for key in self.filedict.keys():
            notthere = []
            for file in self.filedict[key]:
                try:
                    filename=dspath+file
                    file_base = os.path.basename(file)
                    file_save = self.savetopath+'/'+file_base
                    print('\nDownloading',file_base)
                    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
                    filesize = int(req.headers['Content-length'])
                    with open(file_save, 'wb') as outfile:
                        chunk_size=1048576
                        for chunk in req.iter_content(chunk_size=chunk_size):
                            outfile.write(chunk)
                            if chunk_size < filesize:
                                check_file_status(file_save, filesize)
                    check_file_status(file_save, filesize)
                    #print()
                except:
                    notthere.append(file)
            self.filedict[key] = [f for f in self.filedict[key] if f not in notthere]

    def download_retrospective(self,days):
        # JRA data are monthly for all variables after 2014
                
        def check_file_status(filepath, filesize):
            sys.stdout.write('\r')
            sys.stdout.flush()
            size = int(os.stat(filepath).st_size)
            percent_complete = (size/filesize)*100
            sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
            sys.stdout.flush()  
        
        url = 'https://rda.ucar.edu/cgi-bin/login'
        values = {'email' : self.email, 'passwd' : self.password, 'action' : 'login'}
        # Authenticate
        ret = requests.post(url,data=values)
        if ret.status_code != 200:
            print('Bad Authentication')
            print(ret.text)
            exit(1)
        dspath = 'https://rda.ucar.edu/data/ds628.0/'
        
        #daytimes3 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,3)]
        #daytimes6 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,6)]

        #if days[0].day>1:
        #    days.insert(0, dt(days[0].year,days[0].month,1) )     
        if days[-1].day<self._last_day_of_month(days[-1]).day:
            days.insert(-1, dt(days[-1].year,days[-1].month,self._last_day_of_month(days[-1]).day) ) 

        self.days = [d.replace(hour=0,minute=0,second=0,microsecond=0) for d in days]

        tstrt = [d for d in self.days if d.day==1]
        if days[0].day>1:
            tstrt.insert(0, dt(days[0].year,days[0].month,1) )
        tlast = [d for d in days if d.day==self._last_day_of_month(d).day]
        
        # https://rda.ucar.edu/data/ds628.0/anl_p25/2020/anl_p25.007_hgt.2020010100_2020013118  
        # https://rda.ucar.edu/data/ds628.0/anl_surf125/1959/anl_surf125.011_tmp.1959010100_1959123118  
        # https://rda.ucar.edu/data/ds628.0/anl_land125/2020/anl_land125.225_soilw.2020010100_2020013118
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.061_tprat.2019010100_2019013121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.122_shtfl.2019010100_2019013121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.204_dswrf.2019010100_2019013121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.205_dlwrf.2019010100_2019013121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.211_uswrf.2019010100_2019013121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2019/fcst_phy2m125.212_ulwrf.2019010100_2019013121


        phy2mvars = ['061_tprat','122_shtfl','204_dswrf','205_dlwrf','211_uswrf','212_ulwrf']
        surfvars = ['002_prmsl','011_tmp']
        # print(days)

        self.filedict = {\
        'hgt':[f'anl_p125/{ts:%Y}/anl_p125.007_hgt.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'sf':[f'anl_p125/{tstrt[0]:%Y}/anl_p125.035_strm.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast) ],\
        'hgt':[f'anl_p25/{ts:%Y}/anl_p25.007_hgt.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'surf':[f'anl_surf125/{ts:%Y}/anl_surf125.{var}.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast) for var in surfvars],\
        'land':[f'anl_land125/{ts:%Y}/anl_land125.225_soilw.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'phy2m':[f'fcst_phy2m125/{ts:%Y}/fcst_phy2m125.{var}.{ts:%Y%m%d%H}_{tl:%Y%m%d}21' for ts,tl in zip(tstrt,tlast) for var in phy2mvars],\
        'sst':[f'fcst_surf125/{ts:%Y}/fcst_surf125.118_brtmp.{ts:%Y%m%d%H}_{tl:%Y%m%d}21' for ts,tl in zip(tstrt,tlast)]\
                }

        #filelist = [i for j in self.filedict.values() for i in j]
        for key in self.filedict.keys():
            notthere = []
            for file in self.filedict[key]:
                try:
                    filename=dspath+file
                    print(filename)
                    file_base = os.path.basename(file)
                    file_save = self.savetopath+'/'+file_base
                    print('\nDownloading',file_base)
                    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
                    filesize = int(req.headers['Content-length'])
                    with open(file_save, 'wb') as outfile:
                        chunk_size=1048576
                        for chunk in req.iter_content(chunk_size=chunk_size):
                            outfile.write(chunk)
                            if chunk_size < filesize:
                                check_file_status(file_save, filesize)
                    check_file_status(file_save, filesize)
                    #print()
                except:
                    notthere.append(file)
            self.filedict[key] = [f for f in self.filedict[key] if f not in notthere]        

    def download_retrospective_before_2013(self,days):
        print('download_retrospective_monthly')
        # All variable except hgt in JRA are yearly before 2013. Hgt is monthly.
                
        def check_file_status(filepath, filesize):
            sys.stdout.write('\r')
            sys.stdout.flush()
            size = int(os.stat(filepath).st_size)
            percent_complete = (size/filesize)*100
            sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
            sys.stdout.flush()  
        
        url = 'https://rda.ucar.edu/cgi-bin/login'
        values = {'email' : self.email, 'passwd' : self.password, 'action' : 'login'}
        # Authenticate
        ret = requests.post(url,data=values)
        if ret.status_code != 200:
            print('Bad Authentication')
            print(ret.text)
            exit(1)
        dspath = 'https://rda.ucar.edu/data/ds628.0/'
        # dspath = 'https://data.rda.ucar.edu/ds628.0/'# Data path seems to change to this (Jun 14, 2013)
        
        #daytimes3 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,3)]
        #daytimes6 = [d+timedelta(hours=h) for d in self.days for h in range(0,24,6)]

        #if days[0].day>1:
        #    days.insert(0, dt(days[0].year,days[0].month,1) )     
        if days[-1].day<self._last_day_of_month(days[-1]).day:
            days.insert(-1, dt(days[-1].year,days[-1].month,self._last_day_of_month(days[-1]).day) ) 

        self.days = [d.replace(hour=0,minute=0,second=0,microsecond=0) for d in days]

        tstrt = [d for d in self.days if d.day==1]
        if days[0].day>1:
            tstrt.insert(0, dt(days[0].year,days[0].month,1) )
        tlast = [d for d in days if d.day==self._last_day_of_month(d).day]
        
        # https://rda.ucar.edu/data/ds628.0/anl_p25/2020/anl_p25.007_hgt.2020010100_2020013118  
        # https://rda.ucar.edu/data/ds628.0/anl_surf125/1959/anl_surf125.011_tmp.1959010100_1959123118  
        # https://rda.ucar.edu/data/ds628.0/anl_land125/2013/anl_land125.225_soilw.2013010100_2013123118
        # https://data.rda.ucar.edu/ds628.0/anl_land125/2013/anl_land125.225_soilw.2013010100_2013123118
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.061_tprat.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.122_shtfl.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.204_dswrf.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.205_dlwrf.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.211_uswrf.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/fcst_phy2m125/2013/fcst_phy2m125.212_ulwrf.2013010100_2013123121
        # https://rda.ucar.edu/data/ds628.0/anl_p125/2002/anl_p125.035_strm.2002050100_2002053118

        phy2mvars = ['061_tprat','122_shtfl','204_dswrf','205_dlwrf','211_uswrf','212_ulwrf']
        surfvars = ['002_prmsl','011_tmp']

        # Now, creating monthly file names for hgt and yearly file names for others
        self.filedict = {\
        'hgt':[f'anl_p125/{ts:%Y}/anl_p125.007_hgt.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'sf':[f'anl_p125/{tstrt[0]:%Y}/anl_p125.035_strm.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'hgt':[f'anl_p25/{ts:%Y}/anl_p25.007_hgt.{ts:%Y%m%d%H}_{tl:%Y%m%d}18' for ts,tl in zip(tstrt,tlast)],\
        'surf':[f'anl_surf125/{tstrt[0]:%Y}/anl_surf125.{var}.{tstrt[0]:%Y%m%d}00_{tlast[-1]:%Y%m%d}18' for var in surfvars],\
        'land':[f'anl_land125/{tstrt[0]:%Y}/anl_land125.225_soilw.{tstrt[0]:%Y%m%d}00_{tlast[-1]:%Y%m%d}18' ],\
        'phy2m':[f'fcst_phy2m125/{tstrt[0]:%Y}/fcst_phy2m125.{var}.{tstrt[0]:%Y%m%d}00_{tlast[-1]:%Y%m%d}21' for var in phy2mvars],\
        'sst':[f'fcst_surf125/{tstrt[0]:%Y}/fcst_surf125.118_brtmp.{tstrt[0]:%Y%m%d}00_{tlast[-1]:%Y%m%d}21']\
                }
 
        filelist = [i for j in self.filedict.values() for i in j]
        for key in self.filedict.keys():
            notthere = []
            for file in self.filedict[key]:
                try:
                    filename=dspath+file
                    print(filename)
                    file_base = os.path.basename(file)
                    file_save = self.savetopath+'/'+file_base
                    print('\nDownloading',file_base)
                    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
                    filesize = int(req.headers['Content-length'])
                    with open(file_save, 'wb') as outfile:
                        chunk_size=1048576
                        for chunk in req.iter_content(chunk_size=chunk_size):
                            outfile.write(chunk)
                            if chunk_size < filesize:
                                check_file_status(file_save, filesize)
                    check_file_status(file_save, filesize)
                    #print()
                except:
                    notthere.append(file)
            self.filedict[key] = [f for f in self.filedict[key] if f not in notthere]        

    def daily_mean(self,keys=None,days=None,save=True):
        
        if days is None:
            days = self.days
        if keys is None:
            keys = list(self.filedict.keys())      

        self.daily_files={}
        self.available_days={}
        for key in keys:   
            self.daily_files[key]=[]
            self.available_days[key]=[]
            for day in days:
                files = [self.savetopath+'/'+os.path.basename(f) for f in self.filedict[key] if f'{day:%Y%m%d}' in f]
                try:
                    if key == 'phy2m':
                        ds = xr.concat([self._get_colIrr_ds(f) for f in files],dim='time')
                        print(ds)
                    elif key == 'surf':
                        ds1 = xr.concat([self._Pa2hPa(f) for f in files],dim='time')
                        ds2 = xr.open_mfdataset(files,combine='nested',concat_dim='time',engine='cfgrib', \
                                backend_kwargs={'filter_by_keys':{'cfVarName':'t2m'}})
                        ds = xr.merge([ds1,ds2])
                    elif key == 'land':
                        ds = xr.concat([self._soil_layer(f) for f in files],dim='time')
                    else:
                        ds = xr.open_mfdataset(files,combine='nested',concat_dim='time',engine='cfgrib')
                        try:
                            ds.rename({'isobaricInhpa':'level'})
                        except:
                            pass
                    ds_mean = ds.mean(dim='time')
                    ds_mean = ds_mean.expand_dims(dim='time', axis=0)
                    ds_mean.coords['time'] = ('time',[day])
                    if save:
                        ds_mean.to_netcdf(f'{self.savetopath}/{key}_{day:%Y%m%d}.nc')
                    
                    print(key, day)
                    self.daily_files[key].append(f'{self.savetopath}/{key}_{day:%Y%m%d}.nc')
                    self.available_days[key].append(day)
                    
                except:
                    print(f'could not get data for {key} {day:%Y%m%d}')
                    
        #clean up
        savetopath = self.savetopath.replace(' ','\ ')
        try:
            os.system(f'rm {savetopath}/*.idx')
        except:
            pass

    def daily_mean_retrospective(self,keys=None,days=None,save=True):
        
        if days is None:
            days = self.days
        if keys is None:
            keys = list(self.filedict.keys())   
        # print("here we go......")
        self.daily_files={}
        self.available_days={}
        for key in keys:   
            self.daily_files[key]=[]
            self.available_days[key]=[]
            for day in days:
                if day.year <= 2013:
                    if key == 'hgt' or key =='sf':
                        print('we are now processing before 2013 monthly hgt files')
                        files = [self.savetopath+'/'+os.path.basename(f) for f in self.filedict[key] if f'{day:%Y%m}0100' in f]
                    else:  
                        print('we are not processing before 2013 yearly files')
                        files = [self.savetopath+'/'+os.path.basename(f) for f in self.filedict[key]]
                    # The orginal script would creash when processsing not hgt after Feb because 0201 is not in the self.filedict for 2013 and earlier.
                elif day.year > 2013:
                    # This works much faster for 2014 and after.
                    # It only looks for the months need, instead of search every month in the year
                    print('after 2013')
                    files = [self.savetopath+'/'+os.path.basename(f) for f in self.filedict[key] if f'{day:%Y%m}0100' in f]
                #try:
                if key == 'phy2m':
                    print('phy2m')
                    ds_sfc = xr.merge([xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': 'surface'}) for f in files]).sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                    ds_toa = xr.merge([xr.open_dataset(f,engine='cfgrib',filter_by_keys={'typeOfLevel': 'nominalTop'}) for f in files]).sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                    ds = self._get_colIrr(ds_sfc,ds_toa)
                    ds = ds.sel(step=timedelta(hours=3)) # select 3 hour forecast to match real-time 
                elif key == 'surf':
                    print('surf')
                    ds = xr.merge([xr.open_dataset(f,engine='cfgrib') for f in files]).sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                    ds['msl'] = ds['msl']*.01
                elif key == 'land':
                    print('land')
                    ds = xr.merge([xr.open_dataset(f,engine='cfgrib',backend_kwargs={'filter_by_keys':{'cfVarName':'ussl'}}) for f in files]).sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                    ds = ds.sel(threeLayers=slice(1,2)).mean(dim='threeLayers')
                elif key == 'sf':
                    print('sf')
                    ds = xr.open_mfdataset(files,combine='nested',concat_dim='time',engine='cfgrib').sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                elif key == 'sst':
                    print('sst')
                    ds = xr.merge([xr.open_dataset(f,engine='cfgrib',backend_kwargs={'filter_by_keys':{'cfVarName':'btmp'}}) for f in files]).sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))  
                    ds = ds.sel(step=timedelta(hours=3)) # select 3 hour forecast to match real-time, cfVarName = 'btmp' not 'brtmp'
                else:
                    print('hgt')
                    ds = xr.open_mfdataset(files,combine='nested',concat_dim='time',engine='cfgrib').sel(time=str(day.year)+'-'+str(day.month)+'-'+str(day.day))
                    try:
                        # print('change level')
                        ds.DataArray.rename({'isobaricInhPa':'level'})# CYM JRA level name is isobaricInhPa but this doesn't work
                    except:
                        # print('pass change level')
                        pass  
                
                ds_mean = ds.mean(dim='time')
                ds_mean = ds_mean.expand_dims(dim='time', axis=0)
                ds_mean.coords['time'] = ('time',[day])
                if save:
                    ds_mean.to_netcdf(f'{self.savetopath}/{key}_{day:%Y%m%d}.nc')
                
                print(key, day)
                self.daily_files[key].append(f'{self.savetopath}/{key}_{day:%Y%m%d}.nc')
                self.available_days[key].append(day)
                    
                #except:
                #    print(f'could not get data for {key} {day:%Y%m%d}')
                    
        #clean up
        savetopath = self.savetopath.replace(' ','\ ')
        try:
            os.system(f'rm {savetopath}/*.idx')
        except:
            pass   

    def _last_day_of_month(self,any_day):

        # The day 28 exists in every month. 4 days later, it's always next month
        next_month = any_day.replace(day=28) + timedelta(days=4)
        # subtracting the number of the current day brings us back one month
        return next_month - timedelta(days=next_month.day)       

    def _soil_layer(self,filename):
        
        ds = xr.open_dataset(filename, engine='cfgrib',backend_kwargs={'filter_by_keys':{'cfVarName':'ussl'}})
        ds = ds.sel(threeLayers=slice(1,2)).mean(dim='threeLayers')
        return ds

    def _Pa2hPa(self,filename):
        
        ds = xr.open_dataset(filename, engine='cfgrib',backend_kwargs={'filter_by_keys':{'cfVarName':'msl'}})
        ds['msl'] = ds['msl']*.01
        return ds

    def _get_colIrr_ds(self,filename):
        
        # colIrr:
        # (+) Down SW rad flux at nominal top (dswrf)
        # (-) Down SW rad flux at surface (dswrf)
        # (-) Down LW rad flux at surface (dlwrf)
        # (-) Up SW rad flux at nominal top (uswrf)
        # (+) Up SW rad flux at surface (uswrf)
        # (-) Up LW rad flux at nominal top (ulwrf)
        # (+) Up LW rad flux at surface (ulwrf)
        # (+) Up sensible heat flux at surface (shf)
        # (+L*) Precipitation ()
        # 
        # f'fcst_phy2m125/{time-timedelta(days=1):%Y%m}/fcst_phy2m125.{time-timedelta(days=1):%Y%m%d%H}'
        #
        
        ds = xr.open_dataset(filename, engine='cfgrib', \
                             backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface'}})
        
        dsw_sfc = ds['dswrf'] # W m**-2
        dlw_sfc = ds['dlwrf'] # W m**-2
        usw_sfc = ds['uswrf'] # W m**-2
        ulw_sfc = ds['ulwrf'] # W m**-2
        shf = ds['shf'] # W m**-2
        pcp = ds['tpratsfc'] # mm per day
        
        densw = 1e3 # kg m**-3
        day_per_sec = 1/(60*60*24)
        m_per_mm = 1e-3
        pcp = pcp * day_per_sec * m_per_mm * densw # kg m**-2 s**-1
        L = 2.260e6 # J kg**-1
    
        ds = xr.open_dataset(filename, engine='cfgrib', \
                             backend_kwargs={'filter_by_keys':{'typeOfLevel': 'nominalTop'}})
        
        dsw_nt = ds['dswrf'] # W m**-2
        usw_nt = ds['uswrf'] # W m**-2
        ulw_nt = ds['ulwrf'] # W m**-2
        
        colIrr = dsw_nt - dsw_sfc - dlw_sfc - usw_nt + usw_sfc - ulw_nt + ulw_sfc + shf + L*pcp
        
        return colIrr.to_dataset(name='colIrr')

    def _get_colIrr(self,ds_sfc,ds_toa):
        
        # colIrr:
        # (+) Down SW rad flux at nominal top (dswrf)
        # (-) Down SW rad flux at surface (dswrf)
        # (-) Down LW rad flux at surface (dlwrf)
        # (-) Up SW rad flux at nominal top (uswrf)
        # (+) Up SW rad flux at surface (uswrf)
        # (-) Up LW rad flux at nominal top (ulwrf)
        # (+) Up LW rad flux at surface (ulwrf)
        # (+) Up sensible heat flux at surface (shf)
        # (+L*) Precipitation ()
        # 
        # f'fcst_phy2m125/{time-timedelta(days=1):%Y%m}/fcst_phy2m125.{time-timedelta(days=1):%Y%m%d%H}'
        #
        
        #ds = xr.open_dataset(filename, engine='cfgrib', \
        #                     backend_kwargs={'filter_by_keys':{'typeOfLevel': 'surface'}})

        ds = ds_sfc

        dsw_sfc = ds['dswrf'] # W m**-2
        dlw_sfc = ds['dlwrf'] # W m**-2
        usw_sfc = ds['uswrf'] # W m**-2
        ulw_sfc = ds['ulwrf'] # W m**-2
        shf = ds['shf'] # W m**-2
        pcp = ds['tpratsfc'] # mm per day
        
        densw = 1e3 # kg m**-3
        day_per_sec = 1/(60*60*24)
        m_per_mm = 1e-3
        pcp = pcp * day_per_sec * m_per_mm * densw # kg m**-2 s**-1
        L = 2.260e6 # J kg**-1
    
        #ds = xr.open_dataset(filename, engine='cfgrib', \
        #                     backend_kwargs={'filter_by_keys':{'typeOfLevel': 'nominalTop'}})

        ds = ds_toa                     
        
        dsw_nt = ds['dswrf'] # W m**-2
        usw_nt = ds['uswrf'] # W m**-2
        ulw_nt = ds['ulwrf'] # W m**-2
        
        colIrr = dsw_nt - dsw_sfc - dlw_sfc - usw_nt + usw_sfc - ulw_nt + ulw_sfc + shf + L*pcp
        
        return colIrr.to_dataset(name='colIrr')
