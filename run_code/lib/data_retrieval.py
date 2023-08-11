#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:58:51 2020

@author: slillo

Edited: J.R. Albers 6.22.2023
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

# CYM new line
from urllib.request import build_opener # used in CYM new code
# CYM end of new line

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
        # CYM comment out
        # dspath = 'https://rda.ucar.edu/data/ds628.8/'
        # CYM end of comment out

        # CYM new line
        dspath = 'https://data.rda.ucar.edu/ds628.8/'
        # CYM end of new line
        
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
                    
                    # CYM comment out
                    # file_save = self.savetopath+'/'+file_base   
                    # CYM end of comment out
                    
                    # CYM new line
                    file_save = f'./{self.savetopath}/{file_base}'
                    # CYM end of new line

                    print('\nDownloading',file_base)
                    # 6/22/23 CYM commented out because new NCAR RDA page doesn't have 'Content-length' info
                    # CYM comment out
                    # req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
                    # filesize = int(req.headers['Content-length'])
                    # with open(file_save, 'wb') as outfile:
                    #     chunk_size=1048576
                    #     for chunk in req.iter_content(chunk_size=chunk_size):
                    #         outfile.write(chunk)
                    #         if chunk_size < filesize:
                    #             check_file_status(file_save, filesize)
                    # check_file_status(file_save, filesize)
                    # CYM end of comment out
                    #print()
                    
                    # CYM new line
                    opener = build_opener()
                    sys.stdout.flush()
                    infile = opener.open(filename)
                    outfile = open(file_save, "wb")
                    outfile.write(infile.read())
                    outfile.close()
                    sys.stdout.write("done\n")
                    # CYM end of new line
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
