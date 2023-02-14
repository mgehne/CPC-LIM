r"""
Read and prep data for use in the LIM

Sam Lillo, Matt Newman, John Albers

Edited: J.R. Albers 10.4.2022
This function creates variable and EOF data set objects for use in creating and running the NOAA PSL/CPC subseasonal LIM.


"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
from os import listdir
from os.path import isfile, join
import netCDF4 as nc
import numpy as np
import copy
from datetime import datetime as dt,timedelta
import matplotlib as mlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from global_land_mask import globe
from scipy.ndimage import gaussian_filter as gfilt
import statistics
from scipy.interpolate import griddata

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib.tools import *
from lib.plot import PlotMap
from global_land_mask import globe
# from .tools import *
# from .plot import PlotMap

####################################################################################
# MAIN CODE BODY
####################################################################################


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
        self.smoother = kwargs.pop('smoother',None)
        self.coarsegrain = kwargs.pop('coarsegrain',None)
        self.attrs = {}

        # Concatenate all files into one dataset
        filenames = sorted([join(datapath, f) for f in listdir(datapath) \
                     if isfile(join(datapath, f)) and f.endswith('.nc')])

        ds = self.get_ds(filenames)
        if self.climoyears is None:
            self.climoyears = (min(ds['time']).year,max(ds['time']).year)
        climo_set = np.array([(i.year>=min(self.climoyears)) & (i.year<=max(self.climoyears)) for i in ds['time']])
        print(self.climoyears)
        print(ds['time'])
        print(climo_set)
        ds['time'] = ds['time'][climo_set]
        ds['var'] = ds['var'][climo_set]
        ds['var'] = np.where(np.isnan(ds['var']),0,ds['var'])

        self.lat = ds['lat'][self.domain]
        self.lon = ds['lon'][self.domain]
        self.latgrid = ds['lat']
        self.longrid = ds['lon']
        
        # Data manipulation
        if self.climo is None:
            self.climo = get_climo(ds['var'],ds['time'],self.climoyears)
        else:
            self.climo = np.array([self.flatten(i) for i in self.climo])
            self.climo[abs(self.climo)>1e29]=np.nan

        if self.varname == 'anomaly':
            anomaly = copy.copy(ds['var'])
        else:
            anomaly = get_anomaly(ds['var'],ds['time'],self.climo)

        if self.time_window is None:
            self.running_mean = anomaly
        else:
            self.running_mean = get_running_mean(anomaly,self.time_window)[self.time_window:]
            ds['time'] = ds['time'][self.time_window:]

        if season0:
            datewhere = np.where(list(map(self._date_range_test,ds['time'])) & \
                                 (ds['time']>=dt.strptime(f'{min(self.climoyears)}/{self.datebounds[0]}','%Y/%m/%d')) & \
                                 (ds['time']<=dt.strptime(f'{max(self.climoyears)}/{self.datebounds[1]}','%Y/%m/%d')))[0]
        else:
            datewhere = np.where(list(map(self._date_range_test,ds['time'])))[0]

        self.time = ds['time'][datewhere]
        if isinstance(self.time,np.ma.MaskedArray):
            self.time = self.time.data
        self.running_mean = self.running_mean[datewhere]

        self.climo_stdev = np.nanstd(self.running_mean)
        self.climo_mean = np.nanmean(self.running_mean)


    def get_ds(self,filenames):

        ds = {}
        print('--> Starting to gather data')
        timer_start = dt.now()
        for prog,fname in enumerate(filenames):

            ds0 = nc.Dataset(fname)

            if 'climo' in ds0.variables:
                self.climo = ds0['climo']

            lat_name = ([s for s in ds0.variables.keys() if 'lat' in s]+[None])[0]
            lon_name = ([s for s in ds0.variables.keys() if 'lon' in s]+[None])[0]
            lev_name = ([s for s in ds0.variables.keys() if 'lev' in s or 'lv_' in s]+[None])[0]
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

            ds['lat']=ds0[lat_name][:]
            ds['lon']=ds0[lon_name][:]%360
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
            perday = int(86400/(tmp[1]-tmp[0]).total_seconds())
            tmp = tmp[::perday]

            if prog==0:
                ds['time'] = tmp
            else:
                ds['time'] = np.append(ds['time'],tmp)

            if len(ds0[var_name].shape)>3:
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
                lonres = abs(statistics.mode(np.gradient(ds['lon'].data)[1].flatten()))
                latres = abs(statistics.mode(np.gradient(ds['lat'].data)[0].flatten()))
                lonbin = int(self.coarsegrain/lonres)
                latbin = int(self.coarsegrain/latres)
                new_lats = ds['lat'][::latbin,::lonbin]
                new_lons = ds['lon'][::latbin,::lonbin]
                newdata = newdata[:,::latbin,::lonbin]
                ds['lat']=new_lats
                ds['lon']=new_lons

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
                lon_shift = ds['lon'].copy()
                lon_shift[ds['lon']>180] = ds['lon'][ds['lon']>180]-360
                zmask = zmask*globe.is_land(ds['lat'],lon_shift)

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
            update_progress('Gathering data',(prog+1)/len(filenames))
        print('--> Completed gathering data (%.1f seconds)' \
              % (dt.now()-timer_start).total_seconds())
        return ds

    def subset(self,datebounds = ('1/1','12/31'),season0=True):
        self.datebounds = datebounds
        if season0:
            datewhere = np.where(list(map(self._date_range_test,self.time)) & \
                                 (self.time>=dt.strptime(f'{min(self.climoyears)}/{self.datebounds[0]}','%Y/%m/%d')) & \
                                 (self.time<=dt.strptime(f'{max(self.climoyears)}/{self.datebounds[1]}','%Y/%m/%d')))[0]
        else:
            datewhere = np.where(list(map(self._date_range_test,self.time)))[0]

        self.time = self.time[datewhere]
        self.running_mean = self.running_mean[datewhere]

        self.climo_stdev = np.nanstd(self.running_mean)
        self.climo_mean = np.nanmean(self.running_mean)

    def get_latest(self):
        filenames = sorted([join(self.datapath, f) for f in listdir(self.datapath) \
                 if isfile(join(self.datapath, f)) and f.endswith('.nc')])[-2:]
        ds = self.get_ds(filenames)

        anomaly = get_anomaly(ds,self.climo)
        if self.time_window is None:
            running_mean = anomaly
        else:
            running_mean = np.expand_dims(get_running_mean(anomaly,self.time_window)[-1],axis=0)


    def _date_range_test(self,t):
        t_min,t_max = [dt.strptime(i,'%m/%d') for i in self.datebounds]
        t_max += timedelta(days=1,seconds=-1)
        if t_min<t_max:
            test1 = (t>=t_min.replace(year=t.year))
            test2 = (t<=t_max.replace(year=t.year))
            return test1 & test2
        else:
            test1 = (t_min.replace(year=t.year)<=t<dt(t.year+1,1,1))
            test2 = (dt(t.year,1,1)<=t<=t_max.replace(year=t.year))
            return test1 | test2

    def regrid(self,a):
        # Take 1-d vector of same length as domain
        # and transform to original grid
        b = self.mapgrid.copy()
        b[self.domain] = a
        return b

    def flatten(self,a):
        # Take n-d array and flatten
        b = a[self.domain]
        return b

    def plot_map(self,z=None,time=None,ax=None,projection='dynamic',prop={}):

        r"""
        Takes space vector or time, and plots a map of the data.

        Parameters
        ----------
        z : ndarray / list
            Vector with same length as number of space points in self.anomaly
        time : datetime object
            Optional to plot self.anomaly data from specific time in self.time.
        ax : axes instance
            Can pass in own axes instance to plot in existing axes.

        Other Parameters
        ----------------
        prop : dict
            Options for plotting, keywords and defaults include
            * 'cmap' - None
            * 'levels' - negative abs data max to abs data max
            * 'figsize' - (10,6)
            * 'dpi' - 150
            * 'cbarticks' - every other level, determined in get_cmap_levels
            * 'cbarticklabels' - same as cbarticks
            * 'cbar_label' - None. Optional string, or True for self.attrs['units']
            * 'extend' - 'both'
            * 'interpolate' - None. If float value, interpolates to that lat/lon grid resolution.
            * 'drawcountries' - False
            * 'drawstates' - False

        Returns
        -------
        ax : axes instance
        """

        if z is None and time is None:
            z = self.running_mean[-1]
        elif z is None:
            z = self.running_mean[list(self.time).index(time)]

        default_prop={'cmap':None,'levels':None,'fill':True,'figsize':(10,6),'dpi':150,'cbarticks':None,'cbarticklabels':None,\
                      'cbar_label':None,'cbar':True,'extend':'both','interpolate':None,'drawcountries':False,'drawstates':False,\
                      'contour_total':False,'addcyc':False,'res':'m','central_longitude':-90,'latlon':False}
        prop = add_prop(prop,default_prop)

        if prop['cmap'] is None:
            prop['cmap'] = {0:'violet',22.5:'mediumblue',40:'lightskyblue',\
                47.5:'w',52.5:'w',60:'gold',77.5:'firebrick',100:'violet'}
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(z)),np.nanmax(abs(z)))

        if len(prop['levels'])==2:
            mycmap,levels = get_cmap_levels(prop['cmap'],prop['levels'])
        else:
            mycmap,levels = prop['cmap'],prop['levels']

        if prop['cbarticklabels'] is None:
            cbticks = levels[::2]
            cbticklabels = cbticks
        elif prop['cbarticks'] is None:
            cbticks = prop['levels']
            cbticklabels = prop['cbarticklabels']
        else:
            cbticks = prop['cbarticks']
            cbticklabels = prop['cbarticklabels']

        #m,addcyc = map_proj(self.lat,self.lon)

        zmap = self.regrid(z)
        lat = self.latgrid
        lon = self.longrid

        #Fill poles
        try:
            ipole = np.where(lat==90)
            zmap[ipole] = np.nanmean(zmap[tuple([ipole[0]-1,ipole[1]])])
        except:
            pass
        try:
            ipole = np.where(lat==-90)
            zmap[ipole] = np.nanmean(zmap[tuple([ipole[0]+1,ipole[1]])])
        except:
            pass

        if prop['interpolate'] is not None:
            z = self.flatten(zmap)

            xMin = max([0,min(self.lon)-5])
            yMin = max([-90,min(self.lat)-5])
            xMax = min([360,max(self.lon)+5])
            yMax = min([90,max(self.lat)+5])

            grid_res = prop['interpolate']
            xi = np.arange(xMin, xMax+grid_res, grid_res)
            yi = np.arange(yMin, yMax+grid_res, grid_res)
            lon,lat = np.meshgrid(xi,yi)
            # grid the data.
            zLL = z[np.argmin((self.lon-xMin)**2+(self.lat-yMin)**2)]
            zLR = z[np.argmin((self.lon-xMax)**2+(self.lat-yMin)**2)]
            zUL = z[np.argmin((self.lon-xMin)**2+(self.lat-yMax)**2)]
            zUR = z[np.argmin((self.lon-xMax)**2+(self.lat-yMax)**2)]
            lonNew = np.array(list(self.lon)+[xMin,xMax,xMin,xMax])
            latNew = np.array(list(self.lat)+[yMin,yMin,yMax,yMax])
            zNew = np.array(list(z)+[zLL,zLR,zUL,zUR])

            lonNoNan = lonNew[~np.isnan(zNew)]
            latNoNan = latNew[~np.isnan(zNew)]
            zNoNan = zNew[~np.isnan(zNew)]
            zmask = np.where(np.isnan(zNew),1,0)
            zmap = griddata((lonNoNan,latNoNan), zNoNan, (xi[None,:], yi[:,None]), method='cubic')
            zmask = griddata((lonNew,latNew), zmask, (xi[None,:], yi[:,None]), method='linear')
            zmap[zmask>0.9]=np.nan

            zmap = zmap[:,np.where((xi>=min(self.lon)) & (xi<=max(self.lon)))[0]][np.where((yi>=min(self.lat)) & (yi<=max(self.lat)))[0],:]
            xi = xi[np.where((xi>=min(self.lon)) & (xi<=max(self.lon)))]
            yi = yi[np.where((yi>=min(self.lat)) & (yi<=max(self.lat)))]
            lon,lat = np.meshgrid(xi,yi)

        #create figure
        if ax is None:
            fig = plt.figure(figsize = prop['figsize'],dpi=prop['dpi'])
        else:
            # get the figure numbers of all existing figures
            fig_numbers = [x.num for x in mlib._pylab_helpers.Gcf.get_all_fig_managers()]
            # set figure as last figure number
            fig = plt.figure(fig_numbers[-1])

        # Add cyclic
        if len(lon.shape)==1 and len(lat.shape)==1:
            lons,lats = np.meshgrid(lon,lat)
        else:
            lons,lats = lon,lat
        if np.amax(lon)-np.amin(lon)>345:
            lonplt = np.concatenate([lons,lons[:,0][:,None]+360],axis=1)
            latplt = np.concatenate([lats,lats[:,0][:,None]],axis=1)
            dataplt = np.concatenate([zmap,zmap[:,0][:,None]],axis=1)
        else:
            lonplt,latplt,dataplt = lons,lats,zmap

        m = PlotMap(projection,lon=self.lon,lat=self.lat,res=prop['res'],\
                    central_longitude=prop['central_longitude'])
        m.setup_ax(ax=ax)

        if prop['fill']:
            cbmap = m.contourf(lonplt, latplt, dataplt, cmap=mycmap,levels=levels,extend=prop['extend'],zorder=9)
            if prop['interpolate'] is not None and self.landmask:
                m.fill_water(zorder=9)
        else:
            cbmap = ax.contour(lonplt, latplt, dataplt, colors='k',levels=levels)
            if prop['interpolate'] is not None and self.landmask:
                m.fill_water(zorder=9)

        m.drawcoastlines(linewidth=1,color='0.25',zorder=10)
        if prop['drawcountries']:
            m.drawcountries(linewidth=0.5,color='0.25',zorder=10)
        if prop['drawstates']:
            m.drawstates(linewidth=0.5,color='0.25',zorder=10)

        #if m.projection in ['NorthPolarStereo','SouthPolarStereo']:
            #m.stereo_lat_bound()

        if prop['latlon']:
            m.plot_lat_lon_lines(bounds=(np.amin(lonplt),np.amax(lonplt),np.amin(latplt),np.amax(latplt)))

        if prop['fill'] and prop['cbar']:
            #plt.subplots_adjust(bottom=0.12)
            #cax = fig.add_axes([0.15,0.05,0.7,0.03])

            #cbar = fig.colorbar(cbmap,ticks=cbticks,cax=cax,orientation='horizontal')
            cbar = plt.colorbar(cbmap,ticks=cbticks,orientation='horizontal',ax=ax,shrink=.8,fraction=.05,aspect=30,pad=.1)
            cbar.ax.set_xticklabels(cbticklabels)

            if prop['cbar_label'] is not None:
                cbar_label = prop['cbar_label']
                if cbar_label is True:
                    cbar_label = self.attrs['units']
                cbar.ax.set_xlabel(cbar_label,fontsize=14)

        return m.ax


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

            save_ncds(vardict,coords,filename=join(path,f'{self.varlabel}.{K}.nc'))


class eofDataset:

    def __init__(self,varobjs,max_eofs=100,eof_in=None,time_extended=None,skip=1):
        self.time_extended = time_extended
        self.skip = skip
        prepped = []
        if isinstance(varobjs,(tuple,list)):
            self.varobjs = tuple(varobjs)
        else:
            self.varobjs = tuple([varobjs])
        for obj in self.varobjs:
            varstd = (obj.running_mean-obj.climo_mean)/obj.climo_stdev
            tmp = get_area_weighted(varstd,obj.lat)
            tmp = tmp.reshape(tmp.shape[0],np.product(tmp.shape[1:]))
            prepped.append(tmp)
        prepped = np.concatenate(prepped,axis=1)
        if max_eofs > 0:
            if time_extended is None:
                self.eof_dict = get_eofs(prepped,max_eofs,eof_in=eof_in)
            else:
                dims = prepped.shape
                a = prepped.reshape((dims[0],np.product(dims[1:])),order='F')
                prepped_ext = self._extendmat(a,time_extended,skip)
                self.eof_dict = get_eofs(prepped_ext,max_eofs,eof_in=eof_in)
                self.eof_dict['eof'] = self.eof_dict['eof'].reshape((len(self.eof_dict['eof']),*dims[1:],time_extended),order='F')
            self.__dict__.update(self.eof_dict)

    def _extendmat(self,a,per,skip=1):
        a2=np.concatenate([np.ones([per*skip,a.shape[1]])*np.nan,a])
        b=np.ones([int(a.shape[0]//skip),a.shape[1]*per])*np.nan
        for i in range(b.shape[0]): #for each day
            for j in range(per): #for each per
                b[i,j*a.shape[1]:(j+1)*a.shape[1]]=a2[i*skip-j*skip+per*skip,:]
        return np.array(b)

#    def _TEeof(self,testdata,per,howmany):
#        dims = testdata.shape
#        a = testdata.reshape((dims[0],np.product(dims[1:])),order='F')
#        a_ext = self._extendmat(a,per)
#        E,expvar,Z = calceof(a_ext[per-1:],howmany)
#        EOFs = [EE.reshape(*dims[1:],per,order='F') for EE in E.T]
#        return EOFs,expvar,Z

    def get_pc(self,ds,trunc=None):

        if not isinstance(ds,dict):
            print('ds must be a dictionary')

        if len(self.varobjs)==1 and self.varobjs[0].varlabel not in ds.keys():
            ds = {self.varobjs[0].varlabel:ds}

        prepped = []
        for obj in self.varobjs:
            data = ds[obj.varlabel]
            varstd = (data-obj.climo_mean)/obj.climo_stdev
            tmp = get_area_weighted(varstd,obj.lat)
            tmp = tmp.reshape(tmp.shape[0],np.product(tmp.shape[1:]))
            prepped.append(tmp)
        prepped = np.concatenate(prepped,axis=1)

        if trunc is None:
            trunc = prepped.shape[1]
        pc = get_eofs(prepped,eof_in=self.eof[:trunc])

        return pc

    def reconstruct(self,pcs,order=1,num_eofs=None,pc_wt=None):

        r"""
        Method for reconstructing spatial vector from PCs and EOFs

        Parameters
        ----------
        pcs : list or tuple or ndarray
            list of floating numbers corresponding to leading PCs
        order : int
            1 = forecast PCs, 2 = error covariance matrix
        num_eofs : int
            truncation for reconstruction. Default is to retain all EOFs available.
        pc_wt : dict
            keys corresponding to PC number. Values corresponding to weight

        Returns
        -------
        recon : ndarray
            reconstructed spatial vector
        """

        if not isinstance(pcs,np.ndarray):
            pcs = np.array(pcs)
        if len(pcs.shape)<(1+order):
            pcs = pcs.reshape([1]+list(pcs.shape))
        if pc_wt is not None:
            pcs = np.array([pcs[:,i-1]*pc_wt[i] for i in pc_wt.keys()]).squeeze().T
            print(pc_wt)
        if num_eofs is None:
            num_eofs = min(self.eof.shape[0],pcs.shape[-1])
        if order==1:
            recon = np.dot(pcs[:,:num_eofs],self.eof[:num_eofs, :])
        if order==2:
            recon = np.array([np.diag(np.matrix(self.eof).T[:,:num_eofs] @ p \
                           @ np.matrix(self.eof)[:num_eofs,:])**0.5 for p in pcs])

        return_var = {}
        if len(listify(self.varobjs))>1:
            i0 = 0
            for varobj in self.varobjs:
                nlen = varobj.anomaly.shape[1]
                return_var[varobj.varlabel] = recon[:,i0:i0+nlen]*varobj.climo_stdev/np.sqrt(np.cos(np.radians(varobj.lat)))
                i0 += nlen
        else:
            varobj = self.varobjs[0]
            return_var[varobj.varlabel] = recon*varobj.climo_stdev/np.sqrt(np.cos(np.radians(varobj.lat)))

        return return_var

    def plot(self,num_eofs=5,return_figs=False,prop={}):

        r"""
        Map EOFs and plot PC timeseries in subplots.
        Include percent variance explained.

        Parameters
        ----------
        num_eofs : int
            number of EOFs to plot. Default is 4
        return_figs : bool
            Return list of the figures. Default is False

        Returns
        -------
        List of figures
        """

        mycmap=make_colormap({0:'b',.45:'w',.55:'w',1:'r'})

        varobj = self.varobjs[0]
        lat = varobj.latgrid
        lon = varobj.longrid
        PCs = self.eof_dict['pc'][:,:num_eofs].T
        expVARs = self.eof_dict['var_expl_by_eof'][:num_eofs]*100

        TE = self.time_extended

        if TE is None:
            gs = gridspec.GridSpec(2, 1, height_ratios=[7,3])
        else:
            gs = gridspec.GridSpec(TE+1, 1)

        if TE is None:
            TE = 1
            EOFs = np.expand_dims(self.eof_dict['eof'][:num_eofs],-1)
        else:
            EOFs = self.eof_dict['eof'][:num_eofs]
        figs = []
        for iEOF,(EOF,PC,VAR) in enumerate(zip(EOFs,PCs,expVARs)):
            m,addcyc = map_proj(varobj.lat,varobj.lon)
            plt.close()
            fig=plt.figure(figsize=(9,9))

            for iTE in range(TE):
                m._mapboundarydrawn=False
                ax=fig.add_subplot(gs[iTE])

                EOFmap = varobj.regrid(EOF[:,iTE])
                if addcyc:
                    pltdata,lons = addcyclic(EOFmap,lon)
                    lats = np.concatenate([lat,lat[:,0][:,None]],axis=1)
                else:
                    pltdata,lons,lats = EOFmap,lon,lat
                mx,my = m(lons,lats)
                m.drawcoastlines(linewidth=0.5,color='0.5')
                m.drawmapboundary(linewidth=1)

                cs = m.contourf(mx,my,pltdata,np.linspace(-.08,.08,17),cmap=mycmap,extend='both')
                ax.set_title(f'EOF {iEOF+1} (Exp Var={VAR:.1f}%)',size=16)
                cbar = m.colorbar(cs,size='4%')
                cbar.solids.set_edgecolor('face')
                for i in cbar.ax.yaxis.get_ticklabels():
                     i.set_size(16)

            ax2=fig.add_subplot(gs[1+iTE])
            ax2.set_title(f'PC {iEOF+1}',size=16)
            ax2.plot(self.varobjs[0].time,PC)
            yrLoc = mlib.dates.YearLocator(5)
            ax2.xaxis.set_major_locator(yrLoc)
            plt.gca().xaxis.set_major_formatter(mlib.dates.DateFormatter('%Y'))
            ax2.set_ylim([-1.1*np.max(abs(PC)),1.1*np.max(abs(PC))])
            ax2.grid()

            figs.append(fig)
            plt.show()
            plt.close()

        if return_figs:
            return figs

    def save_to_netcdf(self,path):

        vardict = {"pc": {'dims':("time","index"),
                          'data':self.eof_dict['pc'],
                          'attrs':{'long_name':'principal components'}},
                   "varexp": {'dims':("index",),
                              'data':self.eof_dict['var_expl_by_eof'],
                              'attrs':{'long_name':'variance explained by EOF','units':'fraction of 1'}}
                   }
        coords={"index": {'dims':("index",), 'data':np.arange(self.eof_dict['pc'].shape[1])+1},
                "time": {'dims':("time",), 'data':self.varobjs[0].time}}
        eofname = []

        i0 = 0
        for varobj in self.varobjs:
            nlen = varobj.running_mean.shape[1]
            Emap = list(map(varobj.regrid,self.eof_dict['eof'][:,i0:i0+nlen]))
            i0 += nlen

            attrs = copy.copy(varobj.attrs)
            attrs.update({'stdev':varobj.climo_stdev})

            if attrs['units']==None:
                attrs.update({'units':''})
            if attrs['long_name']==None:
                attrs.update({'long_name':''})    

            vardict.update({f"eof_{varobj.varlabel}":
                {'dims':("index",f"lat_{varobj.varlabel}",f"lon_{varobj.varlabel}"),
                 'data':Emap,
                 'attrs':attrs}
                })

            coords.update({f"lon_{varobj.varlabel}": varobj.longrid[0,:],
                           f"lat_{varobj.varlabel}": varobj.latgrid[:,0]})

            coords.update({f"lon_{varobj.varlabel}": {'dims':(f"lon_{varobj.varlabel}",),'data':varobj.longrid[0,:],
                                       'attrs':{'long_name':f'longitude for {varobj.varlabel}','units':'degrees_east'}},
                            f"lat_{varobj.varlabel}": {'dims':(f"lat_{varobj.varlabel}",),'data':varobj.latgrid[:,0],
                                      'attrs':{'long_name':f'latitude for {varobj.varlabel}','units':'degrees_north'}}
                            })
            eofname.append(varobj.varlabel)

        save_ncds(vardict,coords,filename=join(path,f'EOF_{"+".join(eofname)}.nc'))
