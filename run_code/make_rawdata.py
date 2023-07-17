#!/usr/bin/env python
# coding: utf-8

# In[1]:


class varDataset:

    r"""
    Creates an instance of dataset object based on requested files & variable.

    Parameters
    ----------
    path : str
        Directory containing files for concatenation.
    varname : str
        Name of variable in files.
    climo : ndarray


    Other Parameters
    ----------------
    level : float
        If file contains data for multiple levels
    climoyears : tuple
        (start year, end year) to slice data
    latbounds : tuple
        (south lat, north lat) to slice data
    lonbounds : tuple)
        (west lon, east lon) to slice data
    eofs : ndarray
        Provided eofs
    max_eofs : int
        How many modes to retain from the EOF decomposition.
    time_window : int
        Used for running_mean, days.

    Returns
    -------
    Dataset : object
        An instance of Dataset.
    """

    def __init__(self,varlabel,datapath,varname,**kwargs):

        self.varlabel = varlabel
        self.datapath = datapath
        self.varname = varname
        # kwargs
        self.level = kwargs.pop('level',None)
        self.climoyears = kwargs.pop('climoyears',None)
        self.datebounds = kwargs.pop('datebounds',('1/1','12/31'))
        season0 = kwargs.pop('season0',True)
        self.latbounds = kwargs.pop('latbounds',None)
        self.lonbounds = kwargs.pop('lonbounds',None)
        self.time_window = kwargs.pop('time_window',None)
        self.climo = kwargs.pop('climo',None)
        self.landmask = kwargs.pop('landmask',False)
        self.oceanmask = kwargs.pop('oceanmask',False)
        self.smoother = kwargs.pop('smoother',None)
        self.coarsegrain = kwargs.pop('coarsegrain',None)
        self.attrs = {}
        print(varlabel)
        print(datapath)
        print(varname)
        # print(kwargs)
        # Concatenate all files into one dataset
        filenames = sorted([join(datapath, f) for f in listdir(datapath) \
                     if isfile(join(datapath, f)) and f.endswith('.nc')])

        ds = self.get_ds(filenames)
        ds['time'].shape
        if self.climoyears is None:
            print('climoyears are not specified')
            self.climoyears = (min(ds['time']).year,max(ds['time']).year)
        climo_set = np.array([(i.year>=min(self.climoyears)) & (i.year<=max(self.climoyears)) for i in ds['time']])
        # climo_set is the same shape as ds['time'] but has T/F so that the script use this
        # to subset data within the climoyears 
        print(f'climo years = {self.climoyears}')
        print(f'climo_set = {climo_set.shape}')
        # print(f'climo_set = {climo_set}')

        # CYM commented out these two lines. Because the selection of climoyears is done in get_climo,
        # we don't need to subset the desired climoyears. If we have these two lines, then our whole
        # record woud be subset to the same as climoyears. This is not what we want.
        # ds['time'] = ds['time'][climo_set]
        # ds['var'] = ds['var'][climo_set]
        # print(ds['time'].shape)
        # print(ds['time'])

        self.lat = ds['lat'][self.domain]
        self.lon = ds['lon'][self.domain]
        self.latgrid = ds['lat']
        self.longrid = ds['lon']

        # Data manipulation
        if self.climo is None:
            print('getting climo, Line 119 dataset')
            print(ds['var'].shape)
            self.climo = get_climo(ds['var'],ds['time'],self.climoyears)
        else:
            print('self.climo is not None')
            self.climo = np.array([self.flatten(i) for i in self.climo])
            self.climo[abs(self.climo)>1e29]=np.nan

        if self.varname == 'anomaly':
            print('varname has anomaly')
            anomaly = copy.copy(ds['var'])
        else:
            print('getting anomaly, Line 127 dataset')
            anomaly = get_anomaly(ds['var'],ds['time'],self.climo)

        if self.time_window is None:
            print('no time_window!!!!')
            self.running_mean = anomaly
        else:
            print('getting running mean!!!!')
            self.running_mean = get_running_mean(anomaly,self.time_window)[self.time_window:]
            ds['time'] = ds['time'][self.time_window:]
        # CYM:
        # If season0 flag, use only the record in climo years to calculate stddev and mean
        # AND the running_mean, which is output in save_to_netcdf and ultimately used for calculating EOFs,
        # In this case, EOF would be calculated using only the period in climoyears too.
        if season0:
            datewhere = np.where(list(map(self._date_range_test,ds['time'])) & \
                                 (ds['time']>=dt.strptime(f'{min(self.climoyears)}/{self.datebounds[0]}','%Y/%m/%d')) & \
                                 (ds['time']<=dt.strptime(f'{max(self.climoyears)}/{self.datebounds[1]}','%Y/%m/%d')))[0]
            print('in season 0, only use climoyears')
            print(datewhere.shape)
        else:
            print('no season0, keep the whole record for climo')
            datewhere = np.where(list(map(self._date_range_test,ds['time'])))[0]
            print(datewhere.shape)
            

        self.time = ds['time'][datewhere]
        if isinstance(self.time,np.ma.MaskedArray):
            self.time = self.time.data
        self.running_mean = self.running_mean[datewhere]

        self.climo_stdev = np.nanstd(self.running_mean)
        self.climo_mean = np.nanmean(self.running_mean)
        print('we are done here')
        

    def get_ds(self,filenames):

        ds = {}
        print('--> Starting to gather data')
        timer_start = dt.now()
        for prog,fname in enumerate(filenames):
            print(f'getting {fname}')
            ds0 = nc.Dataset(fname)
            # print(ds0.variables.keys())
            if 'climo' in ds0.variables:
                self.climo = ds0['climo']

            lat_name = ([s for s in ds0.variables.keys() if 'lat' in s]+[None])[0]
            lon_name = ([s for s in ds0.variables.keys() if 'lon' in s]+[None])[0]
            lev_name = ([s for s in ds0.variables.keys() if 'isobaricInhPa' in s or 'lv_' in s]+[None])[0]
            time_name = ([s for s in ds0.variables.keys() if 'time' in s]+[None])[0]
            var_name = self.varname

            try:
                self.attrs['long_name']=ds0[var_name].long_name
            except:
                self.attrs['long_name']=None
            try:
                self.attrs['units']=ds0[var_name].units
            except:
                self.attrs['units']=None

            # Convert 'lon' variable from a masked array to a regular NumPy array to use np.where
            # lon_array = np.array(ds0[lon_name][:])
            # lon_array = np.where(lon_array < 0, lon_array +360, lon_array)
            # sorted_indices = np.argsort(lon_array)
            # ds0.variables[lon_name] = lon_array[sorted_indices]
            # ds0.variables[var_name] = ds0.variables[var_name][...,sorted_indices]# lon is the last dimension
            # We don't need to handle longitudes here because they are taken care of during downloading.

            ds['lat']=ds0[lat_name][:]
            ds['lon']=ds0[lon_name][:]            
            lat_original = ds['lat']
            lon_original = ds['lon']
            if len(ds['lat'].shape)==1:
                ds['lon'],ds['lat'] = np.meshgrid(ds['lon'],ds['lat'])
            if lev_name is not None:
                if self.level is None:
                    ilev = 0
                else:
                    ilev = list(ds0[lev_name]).index(self.level)
                    self.attrs['level']=self.level
                ds['lev']=ds0[lev_name][:][ilev]
            try:
                timeunits = ds0[time_name].Units
            except:
                timeunits = ds0[time_name].units
            if timeunits=='Daily':
                yr = int(fname[-7:-3])
                tmp = np.array([dt(yr,1,1)+timedelta(days=i-1) for i in ds0[time_name][:]])
            else:
                tmp = nc.num2date(ds0[time_name][:],timeunits,\
                                  only_use_cftime_datetimes=False,only_use_python_datetimes=True)
                # print(tmp)
            perday = int(86400/(tmp[1]-tmp[0]).total_seconds())
            # print(perday)
            tmp = tmp[::perday]

            if prog==0:
                ds['time'] = tmp
            else:
                ds['time'] = np.append(ds['time'],tmp)

            if len(ds0[var_name].shape)>3:
                print(f'we are selecing level {ilev} here...')
                newdata = ds0[var_name][:,ilev].squeeze()
            elif len(ds0[var_name].shape)<3:
                newdata = ds0[var_name][None,:]
            else:
                newdata = ds0[var_name][:]

            if perday != 1:
                newdata = np.apply_along_axis(lambda x: np.convolve(x,np.ones(perday)/perday, mode='valid')[::4],\
                                              axis=0, arr=newdata)

            if self.smoother is not None:
                newdata = gfilt(newdata,[0]+[self.smoother]*len(newdata.shape[1:]))
            if self.coarsegrain is not None:
                print(f'coarsegraining to {self.coarsegrain} degrees')
                new_lats = np.arange(90,-91,-self.coarsegrain)
                new_lons = np.arange(0,360,self.coarsegrain)
                # print(new_lats)
                # print(new_lons)
                newdata = np.array([interp(lat_original, lon_original, new_lats, new_lons, var_day) for var_day in newdata])
                # lonres = abs(statistics.mode(np.gradient(ds['lon'].data)[1].flatten()))
                # latres = abs(statistics.mode(np.gradient(ds['lat'].data)[0].flatten()))
                # lonbin = int(self.coarsegrain/lonres)
                # latbin = int(self.coarsegrain/latres)
                # new_lats = ds['lat'][::latbin,::lonbin]
                # new_lons = ds['lon'][::latbin,::lonbin]
                # newdata = newdata[:,::latbin,::lonbin]
                
                ds['lon'],ds['lat']= np.meshgrid(new_lons,new_lats)
            
            self.mapgrid = np.ones(newdata.shape[1:])*np.nan

            if self.latbounds is None:
                lim_S = np.amin(ds['lat'])
                lim_N = np.amax(ds['lat'])
            else:
                lim_S = min(self.latbounds)
                lim_N = max(self.latbounds)
            if self.lonbounds is None:
                lim_W = np.amin(ds['lon'])
                lim_E = np.amax(ds['lon'])
            else:
                lim_W = min(self.lonbounds)
                lim_E = max(self.lonbounds)
            zmask = np.ones(self.mapgrid.shape,dtype=bool)
            if self.landmask: 
                print('land masking')
                lon_shift = ds['lon'].copy()
                lon_shift[ds['lon']>180] = ds['lon'][ds['lon']>180]-360
                zmask = zmask*globe.is_land(ds['lat'],lon_shift)

            if self.oceanmask: # For SST
                print('ocean masking')
                lon_shift = ds['lon'].copy()
                lon_shift[ds['lon']>180] = ds['lon'][ds['lon']>180]-360
                zmask = zmask * ~globe.is_land(ds['lat'],lon_shift)
                
            self.domain = np.where((ds['lat']>=lim_S) & \
                              (ds['lat']<=lim_N) & \
                              (ds['lon']>=lim_W) & \
                              (ds['lon']<=lim_E) & \
                              zmask)

            newdata = np.array([n[self.domain] for n in newdata])
            newdata[abs(newdata)>1e29]=np.nan

            if prog==0:
                ds['var'] = newdata
            else:
                ds['var'] = np.append(ds['var'],newdata,axis=0)
        #     update_progress('Gathering data',(prog+1)/len(filenames))
        # print('--> Completed gathering data (%.1f seconds)' \
        #       % (dt.now()-timer_start).total_seconds())
        return ds
    def _date_range_test(self,t):
        t_min,t_max = [dt.strptime(i,'%m/%d') for i in self.datebounds]
        t_max += timedelta(days=1,seconds=-1)
        if t_min<t_max:
            test1 = (t>=t_min.replace(year=t.year))
            test2 = (t<=t_max.replace(year=t.year))
            return test1 & test2
        else:
            # CYM:
            # I think this part is used when the second datebounds is smaller than first
            # Effectively this function can select dates like 3/1-2/28.
            test1 = (t_min.replace(year=t.year)<=t<dt(t.year+1,1,1))
            test2 = (dt(t.year,1,1)<=t<=t_max.replace(year=t.year))
            print('option 2')
            print(test1,test2)
            return test1 | test2
    def regrid(self,a):
        # Take 1-d vector of same length as domain
        # and transform to original grid
        b = self.mapgrid.copy()
        b[self.domain] = a
        return b
        
    def save_to_netcdf(self,path,segmentby=None):

        data_seg = {}
        if segmentby in (None,'all'):
            running_mean = self.running_mean
            time = self.time
            data_seg['all'] = {'running_mean':running_mean,'time':time}

        elif segmentby == 'year':
            years = np.array([t.year for t in self.time])
            for yr in range(min(years),max(years)+1):
                idata = np.where(years==yr)
                running_mean = self.running_mean[idata]
                time = self.time[idata]
                data_seg[yr] = {'running_mean':running_mean,'time':time}

        try:
            attrs = copy.copy(self.attrs)
        except:
            attrs = {}

        for K,V in data_seg.items():
            Vmap = list(map(self.regrid,V['running_mean']))
            Cmap = list(map(self.regrid,self.climo))
            vardict = {"anomaly": {'dims':("time","lat","lon"),
                                   'data':Vmap,
                                   'attrs':attrs},
                       "climo": {'dims':("doy","lat","lon"),
                                 'data':Cmap,
                                 'attrs':attrs}
                       }
            coords={
                "lon": {'dims':('lon',),'data':self.longrid[0,:],
                        'attrs':{'long_name':'longitude','units':'degrees_east'}},
                "lat": {'dims':('lat',),'data':self.latgrid[:,0],
                        'attrs':{'long_name':'latitude','units':'degrees_north'}},
                "time": {'dims':('time',),'data':V['time'],
                         'attrs':{'long_name':'time'}},
                "doy": {'dims':('doy',),'data':np.arange(1,366),
                        'attrs':{'long_name':'day of the year'}},
            }
            print(f'outputting {self.varlabel} to netcdf, Line 545 dataset.py')
            try: 
                os.remove(join(path,f'{self.varlabel}.{K}.nc'))
            except OSError:
                pass
            save_ncds(vardict,coords,filename=join(path,f'{self.varlabel}.{K}.nc'))
    def flatten(self,a):
        # Take n-d array and flatten
        b = a[self.domain]
        return b


# In[2]:


import sys
import numpy as np
import netCDF4 as nc
from datetime import datetime as dt,timedelta
import xarray as xr


# In[3]:

import os
from os import listdir
from os.path import isfile, join


# In[4]:


import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# In[5]:


import lib
# from lib import driver
# from lib import data_retrieval
# from lib.dataset import varDataset
from lib.tools import get_climo
from lib.tools import get_anomaly
from lib.tools import save_ncds
from lib.tools import get_running_mean
from lib.tools import interp


# In[6]:


import statistics
from global_land_mask import globe
import copy



# In[7]:


time_window = 7
datebounds = ('1/1','12/31')
climoyears = (1995,2014)
# climoyears = (1958,2014)
# climoyears = (1995,2014)
# climoyears = (1980,1982)
# climoyears = (1979,2017)
use_vars = {'SST':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/sst','btmp',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/sf','strf',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'SF100':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/sf','strf',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'T2m':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/surf','t2m',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/surf','msl',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'H500':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/hgt','gh',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'H100':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/hgt','gh',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'colIrr':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/phy2m','colIrr',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'SOIL':
                {'info':('/data/ycheng/JRA/Data/make_rawdata_5b_climo-95-14_EOF_58-14/land','ussl',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
                }


# In[8]:

make_vars = ['T2m','SOIL','SLP','colIrr','H500','SST','SF100','SF750']
# make_vars = ['SST','SF100','SF750']
# make_vars = ['SOIL']
#Soil may have a warning due to Nan
#
for name in make_vars:
    out=varDataset(name,*use_vars[name]['info'][:-1],**use_vars[name]['info'][-1])
    dirout_parent = f'/scratch/ycheng/JRA/Data/5b_climo-95-14_EOF_58-14/'
    try: 
        os.mkdir(dirout_parent)
    except OSError:
        pass
    dirout = f'{dirout_parent}/{name}'
    try: 
        os.mkdir(dirout)
    except OSError:
        pass

    try: 
        os.remove(f'{dirout}/{name}_????.nc')
    except OSError:
        pass


    out.save_to_netcdf(dirout,'year')
# Now output is segmented by 'year', no need to output a *all.nc and groupby year anymore.
# CYM old script: outputing all years in one file then groupby year to output yearly files
# data = xr.open_dataset(f'/data/ycheng/JRA/Data/Python/{name}/{name}.all.nc',engine='netcdf4')
# # Group the data by year
# data_grouped = data.groupby('time.year')

# # Iterate over each year and extract the data
# for year, data_year in data_grouped:
#     # Save the data for the current year to a file
#     print(year)
#     filename = f'/data/ycheng/JRA/Data/Python/{name}/{name}.{year}.nc'
#     try:
#         os.remove(f'/data/ycheng/JRA/Data/Python/{name}/{name}.{year}.nc') 
#     except OSError:
#         pass
#     data_year.to_netcdf(filename)
# try: 
#     os.mkdir(f'/data/ycheng/JRA/Data/Python/{name}/all')
#     os.system(f'mv /data/ycheng/JRA/Data/Python/{name}/{name}.all.nc /data/ycheng/JRA/Data/Python/{name}/all')
# except OSError:
#     pass





