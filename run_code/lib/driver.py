r"""
Driver for LIM

Sam Lillo, Matt Newman, John Albers

Edited: J.R. Albers 10.4.2022
This function is the primary driver class for the NOAA PSL/CPC subseasonal LIM.

- Creates gets variables, EOFs, creates LIM models, preps data, runs forecasts (single and blended), saves netCDF files, and calls plotting, etc.

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import pickle
import copy
import os
import sys
import warnings
import imp
import numpy as np
import xarray as xr
import pandas as pd
from scipy.linalg import logm, expm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calendar import monthrange

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib.model import Model
from lib.verif import Verif
from lib.plot import PlotMap
from lib.dataset import varDataset,eofDataset
from lib.tools import *
# from .model import Model
# from .verif import Verif
# from .plot import PlotMap
# from .dataset import varDataset,eofDataset
# from .tools import *


####################################################################################
# MAIN CODE BODY
####################################################################################

class Driver:

    def __init__(self,configFile='namelist.py'):

        r"""
        Read in configuration file with the specifications for variables, EOFs, and the LIM
        """

        #parse configFile string
        tmp = configFile.split('/')
        fname = tmp[-1].split('.')[0]
        fpath = '/'.join(tmp[:-1])

        #import configFile
        fp, path, des = imp.find_module(fname,[fpath])
        namelist = imp.load_module(configFile, fp, path, des)

        #save variables from configFile to driver object
        self.__dict__ = namelist.__dict__
        fp.close()


    def get_variables(self,read=True,save_netcdf_path=None,segmentby=None):

        r"""
        Load data into variable objects and save to pickle files.

        Parameters
        ----------

        read : bool
            if True, reads previously saved variable objects, else compiles data from path

        """

        def add_var(key,vardict):
            self.use_vars[key]=vardict
            self.use_vars[key]['data'] = varDataset(key,*self.use_vars[key]['info'][:-1],**self.use_vars[key]['info'][-1])

        if read:
            for name in self.use_vars.keys():
                self.use_vars[name] = pickle.load( open( f"{self.VAR_FILE_PREFIX}{name}.p", "rb" ) )
        else:
            # Create dataset objects for each variable.
            for name in self.use_vars.keys():
                self.use_vars[name]['data'] = varDataset(name,*self.use_vars[name]['info'][:-1],**self.use_vars[name]['info'][-1])
                pickle.dump(self.use_vars[name], open( f"{self.VAR_FILE_PREFIX}{name}.p", "wb" ) )

        if save_netcdf_path is not None:
            for name in self.use_vars.keys():
                self.use_vars[name]['data'].save_to_netcdf(save_netcdf_path,segmentby)


    def get_eofs(self,read=True,save_netcdf_path=None):

        r"""
        Load data into EOF objects and save to pickle files.

        Parameters
        ----------

        read : bool
            if True, reads previously saved eof objects, else compiles data from path

        """

        self.eofobjs = {}

        if read:
            for limkey,eof_lim in self.eof_trunc_reg.items():
                self.eofobjs[limkey] = {}
                for key in eof_lim.keys():
                    print(f'reading {key} for LIM {limkey}')
                    self.eofobjs[limkey][key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p', "rb" ) )

        if not read:
            # Save EOF objects for each season
            for limkey in self.eof_trunc_reg.keys():
                #Get EOF truncations for the given LIM
                eof_lim = self.eof_trunc_reg[limkey]
                if isinstance(limkey,str):
                    #Then it is a label for full period
                    eof_lim = self.eof_trunc_reg[limkey]
                    tmpobjs = {k:v['data'] for k,v in self.use_vars.items()}

                if isinstance(limkey,int):
                    #Then it is a month
                    #Create copy of variable objects and subset to the given season
                    t1 = (dt(2000,limkey,1)-timedelta(days=0)).strftime('%m/%d')
                    t2 = (dt(2000,limkey,1)+timedelta(days=60)).strftime('%m/%d')
                    tmpobjs = {}
                    for name in self.use_vars.keys():
                        tmp = pickle.load( open( f"{self.VAR_FILE_PREFIX}{name}.p", "rb" ) )
                        tmpobjs[name] = copy.copy(tmp['data'])
                        tmpobjs[name].subset((t1,t2))

                #Calculate EOFs of the subset variable objects
                for key in eof_lim.keys():
                    print(limkey,key)
                    eofobj = eofDataset([tmpobjs[k] for k in listify(key)])
                    pickle.dump(eofobj, open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p','wb'))

        if save_netcdf_path is not None:
            for limkey in self.eof_trunc_reg.keys():
                for key in eof_lim.keys():
                    eofobj = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p', "rb" ) )
                    eofobj.save_to_netcdf(save_netcdf_path+f'_{limkey}')


    def pc_to_grid(self,F=None,E=None,limkey=None,regrid=True,varname=None,fullVariance=False):

        r"""
        Convect PC state space vector to variable grid space.

        Parameters
        ----------
        F : ndarray
            Array with one or two dimensions. LAST axis must be the PC vector.
        E : ndarray
            Array with one or two dimensions. Error covariance matrix. If None, ignores.
            LAST TWO axes must be length of PC vector.
        limkey
            Name of LIM / EOF truncation dictionary. Default is first one in namelist.

        Returns
        -------
        Fmap : dict
            If F was provided. Dictionary with keys as variable names, and values are ndarrays of
            reconstructed gridded space.
        Emap : dict
            If E was provide. Dictionary with keys as variable names, and values are ndarrays of
            reconstructed gridded space.
        """

        if limkey is None:
            limkey = list(self.eof_trunc.keys())[0]
        eof_lim = self.eof_trunc[limkey]

        Fmap, Emap = {}, {}

        if F is not None:
            F = np.asarray(F)
            #Reshape to (times,pcs)
            Pshape = F.shape
            if len(Pshape)==1:
                F = F[np.newaxis,:]
                Pshape = F.shape
            else:
                F = F.reshape((np.product(Pshape[:-1]),Pshape[-1]))
            i0 = 0
            for eofname,plen in eof_lim.items():
                recon = self.eofobjs[limkey][eofname].reconstruct(F[:,i0:i0+plen])
                i0 += plen
                for vname,v in recon.items():
                    if regrid:
                        varobj = self.use_vars[vname]['data']
                        v = np.array(list(map(varobj.regrid,v)))
                        Fmap[vname] = v.reshape(Pshape[:-1]+v.shape[-2:]).squeeze()
                    else:
                        Fmap[vname] = v.reshape((*Pshape[:-1],v.shape[-1])).squeeze()

        if E is not None:
            E = np.asarray(E)
            #Reshape to (times,pcs,pcs)
            Pshape = E.shape
            if len(Pshape)==2:
                E = E[np.newaxis,:,:]
                Pshape = E.shape
            else:
                E = E.reshape((np.product(Pshape[:-2]),*Pshape[-2:]))
            i0 = 0
            for eofname,plen in eof_lim.items():
                eofobj = self.eofobjs[limkey][eofname]
                recon = eofobj.reconstruct(E[:,i0:i0+plen,i0:i0+plen],order=2)
                if fullVariance:
                    truncStdev = {k:np.std(v,axis=0) for k,v in eofobj.reconstruct(eofobj.pc,num_eofs=plen).items()}
                    fullStdev = {v.varlabel:np.std(v.running_mean,axis=0) for v in eofobj.varobjs}
                    varScaling = {k:fullStdev[k]/truncStdev[k] for k in fullStdev.keys()}
                else:
                    varScaling = {v.varlabel:np.ones(v.lon.shape) for v in eofobj.varobjs}

                i0 += plen
                for vname,v in recon.items():
                    if regrid:
                        varobj = self.use_vars[vname]['data']
                        v = np.array(list(map(varobj.regrid,v*varScaling[vname])))
                        Emap[vname] = v.reshape(Pshape[:-2]+v.shape[-2:]).squeeze()
                    else:
                        Emap[vname] = v.reshape((*Pshape[:-2],v.shape[-1])).squeeze()*varScaling[vname]

        if E is None and F is None:
            print('both F and E inputs were None')
            return None

        if varname is None:
            out = tuple([x for x in (Fmap,Emap) if len(x)>0])
        else:
            out = tuple([x[varname] for x in (Fmap,Emap) if len(x)>0])

        if len(out)>1:
            return out
        else:
            return out[0]


    def pc_to_pc(self,pcin,var1=None,var2=None,limkey=None):

        r"""
        Compute the regression matrix to regress var2 (to var) onto var1 (from var). The var1 string should
        be the same variable the pcin is based on.  The regression coefficients are computed from the clima-
        tological LIM EOF/PC values for the input variables. The coefficients are then applied to the current 
        forecast PCs for var1.
        Parameters
        ----------
        pcin : ndarray
            Array with one or two dimensions. LAST axis must be the PC vector.
        var1 : string
            Variable name of input PC.
        var2 : string  
            Variable name of output PC. 
        limkey
            Name of LIM / EOF truncation dictionary. Default is first one in namelist.

        Returns
        -------
        pcout : ndarray
            PC of var2 regressed onto var1.
        """

        if limkey is None:
            limkey = list(self.eof_trunc_reg.keys())[0]
        eof_lim = self.eof_trunc_reg[limkey]

        eof1 = self.eofobjs[limkey][var1]
        eof2 = self.eofobjs[limkey][var2]

        time1,pc1 = (eof1.varobjs[0].time,eof1.pc)
        time2,pc2 = (eof2.varobjs[0].time,eof2.pc)

        frompc = {pd.to_datetime(t):p for t,p in zip(time1,pc1)}
        topc = {pd.to_datetime(t):p for t,p in zip(time2,pc2) if pd.to_datetime(t) in frompc.keys()}
        frompc = {t:p for t,p in frompc.items() if t in topc.keys()}

        stacked = np.array([[frompc[t][:5],topc[t][:5]] for t in frompc.keys()]).swapaxes(0,1)

        C0 = np.matmul(stacked[0].T, stacked[0]) / (stacked[0].shape[0] - 1)
        Ctau = np.matmul(stacked[1].T, stacked[0]) / (stacked[1].shape[0] - 1)

        G = np.matmul(Ctau, np.linalg.pinv(C0))

        pcout = np.matmul(G, np.matrix(pcin).T).T
        return pcout


    def cross_validation(self,limkey=None,num_folds=10,lead_times=np.arange(1,29),average=False,\
                         togrid=False,fullVariance=False,save_netcdf_path=None,segmentby='runtime'):

        r"""
        Cross validation.

        Parameters
        ----------
        num_folds : int
            Number of times data is subset and model is trained. Default is 10,
            Which means each iteration trains on 90% of data, forecasts run on 10% of the data.
        lead_times : list, tuple, ndarray
            Lead times (in days) to integrate model for output
        save_netcdf_path : str
            Path to save netcdf files containing forecast and spread
        segmentby : str
            Only applies for when saving to netcdf. Default to "runtime".

        Returns
        -------
        model_F : dict
            Dictionary of model forecast output, with keys corresponding to valid time
        model_E : dict
            Dictionary of model error output, with keys corresponding to valid time
        """

        #run cross-validation with specified state space
        self.lead_times = lead_times
        self.model_F = {}
        self.model_E = {}
        if togrid:
            self.F_recon,self.E_recon = {},{}
        if limkey is None and all([isinstance(k,int) for k in self.eof_trunc.keys()]):
            self.limkey = '_monthly_'
            for mn in self.eof_trunc.keys():
                print(mn)
                eof_lim = self.eof_trunc[mn]
                eofobjs={}
                for key in eof_lim.keys():
                    eofobjs[key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{mn}.p', "rb" ) )

                V = Verif(eofobjs,eof_lim)
                V.kfoldval(lead_times = lead_times, k = num_folds, tau1n = self.tau1n, average=average)

                F={k:v for k,v in V.fcsts.items() if k.month==mn}
                E={k:v for k,v in V.variance.items() if k.month==mn}
                for t,f in F.items():
                    self.model_F[t] = f
                    self.model_E[t] = E[t]
                if togrid:
                    print('pc to grid')
                    Fr,Er = self.pc_to_grid(F=list(F.values()),E=list(E.values()),\
                                varname='T2m',limkey=mn,regrid=False,fullVariance=fullVariance)
                    for t,f,e in zip(list(F.keys()),list(Fr.values()),list(Er.values())):
                        self.F_recon[t] = f
                        self.E_recon[t] = e

        else:
            if limkey is None:
                limkey = list(self.eof_trunc.keys())[0]
            self.limkey = limkey
            print(limkey)
            eof_lim = self.eof_trunc[limkey]
            eofobjs={}
            for key in eof_lim.keys():
                eofobjs[key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p', "rb" ) )

            V = Verif(eofobjs,eof_lim)
            V.kfoldval(lead_times = lead_times, k = num_folds, tau1n = self.tau1n, average=average)

            for t in V.fcsts.keys():
                self.model_F[t] = V.fcsts[t]
                self.model_E[t] = V.variance[t]


        if save_netcdf_path is not None:
            # Save model attributes to netcdf file
            if not isinstance(save_netcdf_path,str):
                raise TypeError("save_to_netcdf must be a string containing path and filename for netcdf file.")

            else:

                data_seg = {}
                if segmentby == 'runtime':
                    for (K,F),(_,E) in zip(self.model_F.items(),self.model_E.items()):

                        # F and E are a 2darray, lt x pc
                        # pc are concatenated PCs from each EOF

                        if self.limkey == '_monthly_':
                            eofobjs={}
                            eof_lim = self.eof_trunc[K.month]
                            for key in eof_lim.keys():
                                eofobjs[key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{K.month}.p', "rb" ) )
                        else:
                            eofobjs={}
                            for key in eof_lim.keys():
                                eofobjs[key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p', "rb" ) )

                        vardict = {}
                        coords={"leadtime": {'dims':('leadtime',),
                                             'data':self.lead_times,
                                             'attrs':{'long_name':'lead time','units':'days'}}
                                }

                        i0 = 0
                        for eofname,plen in eof_lim.items():

                            recon = eofobjs[eofname].reconstruct(F[i0:i0+plen])
                            i0 += plen

                            if not isinstance(recon,dict):
                                recon = {eofname:recon}

                            for varname,v in recon.items():
                                Vmap = list(map(self.use_vars[varname]['data'].regrid,v))

                                vardict.update({f"{varname}": {'dims':("leadtime",f"lat_{varname}",f"lon_{varname}"),
                                                               'data':Vmap,
                                                               'attrs':self.use_vars[varname]['data'].attrs}
                                                })

                                coords.update({f"lon_{varname}": {'dims':(f"lon_{varname}",),
                                                                  'data':varobj.longrid[0,:],
                                                                  'attrs':{'long_name':f'longitude for {varname}','units':'degrees_east'}},
                                               f"lat_{varname}": {'dims':(f"lat_{varname}",),
                                                                  'data':varobj.latgrid[:,0],
                                                                  'attrs':{'long_name':f'latitude for {varname}','units':'degrees_north'}}
                                               })

                        save_ncds(vardict,coords,filename=join(path,f'{K:%Y%m%d}.nc'))



    def plot_acc(self,varname,lead_time,year_range=None,date_range=None,rhoinf_ptile_range=None,prop={}):

        if year_range is None:
            year_range = (min(self.model_F.keys()).year,max(self.model_F.keys()).year)
        if date_range is None:
            date_range = ('1/1','12/31')

        def lt_avg(v):
            if isinstance(lead_time,(int,float)):
                ilt = [i for i,j in enumerate(self.lead_times) if j==lead_time]
                return v[ilt].squeeze()
            else:
                ilt = [i for i,j in enumerate(self.lead_times) if j in lead_time]
                return np.mean(v[ilt],axis=0)

        model_F = {k:lt_avg(v) for k,v in self.model_F.items() if date_range_test(k,date_range,year_range)}
        model_E = {k:lt_avg(v) for k,v in self.model_E.items() if date_range_test(k,date_range,year_range)}

        if self.limkey != '_monthly_':
            eof_lim = self.eof_trunc[self.limkey]
            varpci = get_varpci(eof_lim,varname)
            # Load EOF object for verif_var for given limkey
            varobj = self.use_vars[varname]['data']
            eofobj = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(varname))+f'_{self.limkey}.p', "rb" ) )

            # Isolate PC for varname, for forecast vector and error covariance matrix, by time
            F_verif = {t:model_F[t][min(varpci):max(varpci)] for t in model_F.keys()}
            E_verif = {t:model_E[t][min(varpci):max(varpci),min(varpci):max(varpci)] for t in model_E.keys()}

            # Reconstruct into grid space, forecast and spread
            F_verif = {t:eofobj.reconstruct(f).squeeze() for t,f in F_verif.items()}
            S_verif = {t:np.diag(np.matrix(eofobj.eof_dict['eof']).T[:,:eof_lim[varname]] \
                                 @ E_verif[t] @ np.matrix(eofobj.eof_dict['eof'])[:eof_lim[varname],:])**0.5\
                      for t in E_verif.keys()}

            # Get observed values for verification
            if len(listify(lead_time))==1:
                O_time_init = np.array(varobj.time)-timedelta(days=lead_time)
                O_data = varobj.running_mean
            # elif multiple lead-times are listed for averaging, must average verification
            elif len(listify(lead_time))>1:

                xlt = max(lead_time)
                nlt = min(lead_time)

                O_time_init = (np.array(varobj.time)-timedelta(days=xlt))[xlt-nlt:]

                O_data = np.mean([varobj.running_mean[lt-nlt:len(varobj.running_mean)+lt-xlt]\
                                                      for lt in lead_time],axis=0)

            # Make dictionary for observations, INIT time : VERIF data
            O_verif = {t:o for t,o in zip(O_time_init,O_data) if t in F_verif.keys()}

            LAC = calc_lac(list(F_verif.values()),list(O_verif.values()))

            #Set default properties
            default_cmap = {0:'violet',
                          .22:'mediumblue',
                          .35:'lightskyblue',
                          .44:'w',
                          .56:'w',
                          .65:'gold',
                          .78:'firebrick',
                          1:'violet'}
            default_title = f'{varname} {lead_time}-day ACC | {min(year_range)} – {max(year_range)} | {" – ".join(date_range)}'
            default_prop={'cmap':default_cmap,'levels':(-1*max(abs(LAC)),max(abs(LAC))),'title':default_title,\
                          'figsize':(10,6),'dpi':150,'drawcountries':True,'drawstates':True}
            prop = add_prop(prop,default_prop)
            prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])
            ax=varobj.plot_map(LAC,prop=prop)
            ax.set_title(prop['title'])


    def get_model(self,limkey,load_file=None,save_file=None,save_to_netcdf=None):

        r"""
        Get the LIM.

        Parameters
        ----------
        month : int
            month of data to train LIM on.
        load_file : str
            Filename for pickle file containing model to load.
            Default is None, in which case a new model is trained.
        save_file : str
            Filename of pickle file to save new model to.
            Default is None, in which case the model is not saved.

        Returns
        -------
        model : model object
            Object of trained model
        """

        if load_file is None:

            self.limkey = limkey

            eof_lim = self.eof_trunc[limkey]
            eofobjs={}
            for key in eof_lim.keys():
                if eof_lim[key]>0:
                    eofobjs[key] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(key))+f'_{limkey}.p', "rb" ) )

            varobjs = [v for name in eofobjs.keys() for v in eofobjs[name].varobjs]

            times = varobjs[0].time

            p = {}
            for name in eofobjs.keys():
                time = eofobjs[name].varobjs[0].time
                pcs = eofobjs[name].eof_dict['pc'][:,:eof_lim[name]]
                p[name] = [pc for t,pc in zip(time,pcs) if t in times]

            all_data = np.concatenate([p[name] for name in p.keys()],axis=1)

            # Get times for tau0 and tau1 data by finding the intersection of all
            # times and times + tau1n days
            times1 = np.intersect1d(times,times+timedelta(days = self.tau1n))
            times0 = times1-timedelta(days = self.tau1n)

            # Get tau0 and tau1 data by taking all data and all times and matching
            # with the corresponding times for tau0 and tau1
            tau0_data = np.array([d for d,t in zip(all_data,times) if t in times0])
            tau1_data = np.array([d for d,t in zip(all_data,times) if t in times1])

            # Train the model
            self.model = Model(tau0_data,tau1_data,tau1n=self.tau1n)
            if save_file is not None:
                pickle.dump(self.model, open(save_file,'wb'))

        else:
            # Read in model
            self.model = pickle.load( open(load_file, "rb" ) )

        if save_to_netcdf is not None:
            # Save model attributes to netcdf file
            if isinstance(save_to_netcdf,str):
                self.model.save_to_netcdf(save_to_netcdf)
            else:
                raise TypeError("save_to_netcdf must be a string containing path and filename for netcdf file.")


    def prep_realtime_data(self,limkey,verbose=False):

        r"""
        Compile realtime data, interpolate to same grid as LIM, and convert into PCs.
        """

        if 'time' not in self.RT_VARS.keys():

            for name in self.RT_VARS.keys():
                if verbose:print(f'reading {name}')
                ds = nc.Dataset(self.RT_VARS[name]['filename'])

                times = nc.num2date(ds['time'][:],ds['time'].units,\
                            only_use_cftime_datetimes=False,only_use_python_datetimes=True)
                perday = int(86400/(times[1]-times[0]).total_seconds())
                times = times[::perday]

                newdata = ds[self.RT_VARS[name]['varname']][:]

                if 'level' in self.RT_VARS[name].keys():
                    levs = ds[self.RT_VARS[name]['levname']][:]
                    level_for_var = np.where(levs==self.RT_VARS[name]['level'])[0]
                    newdata = newdata[:,level_for_var].squeeze()

                newdata = np.apply_along_axis(lambda x: np.convolve(x,np.ones(perday)/perday, mode='valid')[::perday],\
                                                              axis=0, arr=newdata)
                running_mean = get_running_mean(newdata,7)[7:]

                self.RT_VARS[name]['var'] = running_mean
                self.RT_VARS[name]['lat'] = ds['latitude'][:]
                self.RT_VARS[name]['lon'] = ds['longitude'][:]
                self.RT_VARS[name]['time'] = times[7:]

            # find all common times
            p = [v['time'] for v in self.RT_VARS.values()]
            common_times = set(p[0]).intersection(*p)
            for name,v in self.RT_VARS.items():
                if verbose:print(f'common times {name}')
                ikeep = np.array(sorted([list(v['time']).index(j) for j in common_times]))
                self.RT_VARS[name]['var'] = v['var'][ikeep]
                self.RT_VARS[name]['time'] = v['time'][ikeep]

            # interpolate to LIM variable grids
            RT_INTERP = {}
            for name in self.RT_VARS.keys():
                if verbose:print(f'interp {name}')
                data = self.RT_VARS[name]['var']
                data[np.isnan(data)] = 0
                RT_INTERP[name] = np.array([interp2LIM(self.RT_VARS[name]['lat'],self.RT_VARS[name]['lon'],\
                                           var_day,self.use_vars[name]['data']) for var_day in data])

            self.RT_ANOM = {}
            for name in self.RT_VARS.keys():
                if verbose:print(f'anom {name}')
                self.RT_ANOM[name] = get_anomaly(RT_INTERP[name],self.RT_VARS[name]['time'],\
                                                 self.use_vars[name]['data'].climo)

            self.RT_VARS['time'] = self.RT_VARS[name]['time']

        self.RTLIMKEY = limkey

        eof_lim = self.eof_trunc[limkey]
        eofobjs = self.eofobjs[limkey]

        self.RT_PCS = {}
        for name in self.RT_ANOM.keys():
            prepped = get_area_weighted(self.RT_ANOM[name],self.use_vars[name]['data'].lat)
            prepped = prepped / self.use_vars[name]['data'].climo_stdev
            pc = get_eofs(prepped,eof_in=eofobjs[name].eof_dict['eof'][:eof_lim[name]])
            self.RT_PCS[name] = pc


    def run_forecast(self,t_init=None,lead_times=np.arange(1,29),fullVariance=False,save_netcdf_path=None):

        r"""
        Run forecasts with LIM model.

        Parameters
        ----------
        t_init : datetime object
            Date of initialization for the forecast
        lead_times : list, tuple, or ndarray
            lead_times in increment of data to integrate model forward

        Returns
        -------
        model_F : dict
            Dictionary of model forecast output, with keys corresponding to valid time
        model_E : dict
            Dictionary of model error output, with keys corresponding to valid time

        """

        limkey = self.RTLIMKEY

        self.lead_times = [int(i) for i in lead_times]

        if t_init is None:
            t_init = max(self.RT_VARS['time'])

        init_times = [t_init+timedelta(days=i-6) for i in range(7)]
        init_times = [t for t in init_times if t in self.RT_VARS['time']]

        self.get_model(limkey=self.RTLIMKEY)

        # Run the model

        # Get the (time independent) variance from the model
        C0 = np.matrix(self.model.C0)
        Gtau = {lt:expm(np.matrix(self.model.L)*lt) for lt in lead_times}
        Etau = {lt:(C0 - Gtau[lt] @ C0 @ Gtau[lt].T) for lt in lead_times}

        eof_lim = self.eof_trunc[self.RTLIMKEY]
        init_data = np.concatenate([[p for t,p in zip(self.RT_VARS['time'],self.RT_PCS[name]) if t in init_times]\
                                        for name in eof_lim.keys()],axis=1)

        fcst = self.model.forecast(init_data,lead_time=lead_times)

        fcst = np.array(fcst).swapaxes(0,1)
        variance = np.array([Etau[lt] for lt in lead_times])

        self.model_F = {}
        self.model_E = {}
        for i,t in enumerate(init_times):
            self.model_F[t] = fcst[i]
            self.model_E[t] = variance

        self.F_recon = {}
        self.E_recon = {}
        for i,t in enumerate(init_times):
            self.F_recon[t],self.E_recon[t] = self.pc_to_grid(F=fcst[i],E=variance,\
                                limkey=self.RTLIMKEY,regrid=False,fullVariance=fullVariance)

        if save_netcdf_path is not None:
            print('making forecast netcdf files')

            if t_init is None:
                t_init = max(self.model_F.keys())

            if t_init not in self.model_F.keys():
                sys.exit()

            init_times = [t_init+timedelta(days=i-6) for i in range(7)]
            init_times = [t for t in init_times if t in self.model_F.keys()]
            F = [self.model_F[t] for t in init_times]
            E = [self.model_E[t] for t in init_times]
            Fmap,Emap = self.pc_to_grid(F=F,E=E,limkey=self.RTLIMKEY,regrid=True,fullVariance=fullVariance)

            for varname in Fmap.keys():
                print(f'making {varname} netcdf file')
                varobj = self.use_vars[varname]['data']

                coords = {"time": {'dims':('time',),
                                     'data':np.array(init_times),
                                     'attrs':{'long_name':'initial time',}},
                        "lead_time": {'dims':('lead_time',),
                                     'data':self.lead_times,
                                     'attrs':{'long_name':'lead time','units':'days'}},
                        "lon": {'dims':("lon",),
                                  'data':varobj.longrid[0,:],
                                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                        "lat": {'dims':("lat",),
                                  'data':varobj.latgrid[:,0],
                                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                        }

                vardict = {f"{varname}": {'dims':("time","lead_time","lat","lon"),
                                               'data':Fmap[varname],},
                           f"{varname}_spread": {'dims':("time","lead_time","lat","lon"),
                                               'data':Emap[varname],}
                                }

                save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{init_times[-1]:%Y%m%d}.nc'))


    def run_forecast_blend(self,limkey=None,t_init=None,lead_times=np.arange(1,29),fullVariance=True,pc_convert=None,save_netcdf_path=None):

        r"""
        Run forecasts with LIM model.

        Parameters
        ----------
        t_init : datetime object
            Date of initialization for the forecast
        lead_times : list, tuple, or ndarray
            lead_times in increment of data to integrate model forward

        Returns
        -------
        model_F : dict
            Dictionary of model forecast output, with keys corresponding to valid time
        model_E : dict
            Dictionary of model error output, with keys corresponding to valid time

        """

        self.lead_times = [int(i) for i in lead_times]

        if t_init is None:
            t_init = max(self.RT_VARS['time'])

        init_times = [t_init+timedelta(days=i-6) for i in range(7)]
        init_times = [t for t in init_times if t in self.RT_VARS['time']]

        print(init_times)

        mn2 = t_init.month
        mn1 = (mn2-2)%12+1
        mn3 = (mn2)%12+1

        fcsts = {}
        for m in [mn1,mn2,mn3]:
            self.prep_realtime_data(limkey = m)
            eof_lim = self.eof_trunc[m]
            init_data = np.concatenate([[p for t,p in zip(self.RT_VARS['time'],self.RT_PCS[name]) if t in init_times]\
                                        for name in eof_lim.keys() if name in self.RT_PCS.keys()],axis=1)
            self.get_model(limkey = m)
            fcst = self.model.forecast(init_data,lead_time=lead_times)
            print(f'Got Forecast From LIM {m}')

            # Get the (time independent) variance from the model
            C0 = np.matrix(self.model.C0)
            Gtau = {lt:expm(np.matrix(self.model.L)*lt) for lt in lead_times}
            Etau = {lt:(C0 - Gtau[lt] @ C0 @ Gtau[lt].T) for lt in lead_times}

            fcst = np.array(fcst).swapaxes(0,1)
            variance = np.array([Etau[lt] for lt in lead_times])

            if pc_convert is not None:
                i1,i2 = get_varpci(self.eof_trunc[m],pc_convert[0])
                
                for i,f in enumerate(fcst):
                    pcin = np.squeeze(fcst[i,:,i1:i2])
                    out = self.pc_to_pc(pcin,var1=pc_convert[0],var2=pc_convert[1],limkey=m)
                    f[:,i1:i2] = out
                    fcst[i] = f   

            F = {}
            E = {}
            for i,t in enumerate(init_times):
                F[t],E[t] = self.pc_to_grid(F=fcst[i],E=variance,\
                                limkey=m,regrid=False,fullVariance=fullVariance)
            fcsts[m] = {'F':F,'E':E}

        days_in_month = max(monthrange(t_init.year,t_init.month))
        weights = {mn1:1-min([t_init.day,7])/7,mn2:1,mn3:1-min([days_in_month-t_init.day,7])/7}

        self.F_recon = {}
        self.E_recon = {}
        for i,t in enumerate(list(F.keys())):
            self.F_recon[t] = {varname:sum([weights[m]*f['F'][t][varname] for m,f in fcsts.items()]) / sum(weights.values()) for varname in F[t].keys()}
            self.E_recon[t] = {varname:sum([weights[m]*f['E'][t][varname] for m,f in fcsts.items()]) / sum(weights.values()) for varname in F[t].keys()}

        if save_netcdf_path is not None:

            if t_init is None:
                t_init = max(self.F_recon.keys())

            if t_init not in self.F_recon.keys():
                print(f'{t_init:%Y%m%d} NOT IN FORECAST KEYS')
                sys.exit()

            Fmap = {varname:[[self.use_vars[varname]['data'].regrid(f) for f in F[varname]]\
                             for t,F in self.F_recon.items() if t in init_times]\
                    for varname in self.F_recon[t_init].keys()}
            Emap = {varname:[[self.use_vars[varname]['data'].regrid(e) for e in E[varname]]\
                             for t,E in self.E_recon.items() if t in init_times]\
                    for varname in self.E_recon[t_init].keys()}

            for varname in Fmap.keys():
                varobj = self.use_vars[varname]['data']

                coords = {"time": {'dims':('time',),
                                     'data':np.array(init_times),
                                     'attrs':{'long_name':'initial time',}},
                        "lead_time": {'dims':('lead_time',),
                                     'data':self.lead_times,
                                     'attrs':{'long_name':'lead time','units':'days'}},
                        "lon": {'dims':("lon",),
                                  'data':varobj.longrid[0,:],
                                  'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                        "lat": {'dims':("lat",),
                                  'data':varobj.latgrid[:,0],
                                  'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                        }

                vardict = {f"{varname}": {'dims':("time","lead_time","lat","lon"),
                                               'data':Fmap[varname],},
                           f"{varname}_spread": {'dims':("time","lead_time","lat","lon"),
                                               'data':Emap[varname],}
                                }

                save_ncds(vardict,coords,filename=os.path.join(save_netcdf_path,f'{varname}.{init_times[-1]:%Y%m%d}.nc'))


    def save_netcdf_files(self,varname='T2m',t_init=None,lead_times=None,save_to_path=None,add_offset=None,average=False,append_name=None):

        lead_times = listify(lead_times)
        ilt = np.array([self.lead_times.index(l) for l in lead_times])

        FMAP = self.F_recon[t_init][varname][ilt]
        SMAP = self.E_recon[t_init][varname][ilt]

        varobj = self.use_vars[varname]['data']

        if add_offset is not None:
            print('getting offset')
            ds = xr.open_dataset(add_offset)
            days = [int(f'{t_init+timedelta(days = lt):%j}') for lt in lead_times]
            newclim = np.mean([ds[varname].data[d-1] for d in days],axis=0)
            oldclim = np.mean([varobj.climo[d-1] for d in days],axis=0)

            diff = oldclim-newclim
            prepped = get_area_weighted(diff,varobj.lat)
            prepped = prepped / varobj.climo_stdev
            eof_lim = self.eof_trunc[t_init.month]
            eofobjs = self.eofobjs[t_init.month]
            pc = get_eofs(prepped,eof_in=eofobjs[varname].eof_dict['eof'][:eof_lim[varname]])
            diffrecon = eofobjs[varname].reconstruct(pc)[varname]

            FMAP = FMAP+diffrecon.squeeze()
            print(f'RMSE: {np.mean(diffrecon**2)**.5}')


        # make probabilistic forecast map
        bounds = [-np.inf*np.ones(FMAP.shape[-1]),np.zeros(FMAP.shape[-1]),np.inf*np.ones(FMAP.shape[-1])]
        cat_fcst = [get_categorical_fcst((F,),(S,),bounds)[0][1]*100 for F,S in zip(FMAP,SMAP)]

        anom = [varobj.regrid(F) for F in FMAP]
        spread = [varobj.regrid(S) for S in SMAP]
        prob = [varobj.regrid(F) for F in cat_fcst]

        if average:
            anom = [np.mean(anom,axis=0)]
            spread = [np.mean(spread,axis=0)]
            prob = [np.mean(prob,axis=0)]

        g = np.isnan(anom[0])
        anom = [a[:, ~np.all(g, axis=0)][~np.all(g, axis=1)] for a in anom]
        spread = [s[:, ~np.all(g, axis=0)][~np.all(g, axis=1)] for s in spread]
        prob = [p[:, ~np.all(g, axis=0)][~np.all(g, axis=1)] for p in prob]
        lon = varobj.longrid[:, ~np.all(g, axis=0)][~np.all(g, axis=1)][0,:]
        lat = varobj.latgrid[:, ~np.all(g, axis=0)][~np.all(g, axis=1)][:,0]

        coords = {"time": {'dims':('time',),
                             'data':np.array([t_init]),
                             'attrs':{'long_name':'initial time',}},
                "lead_time": {'dims':('lead_time',),
                             'data':lead_times,
                             'attrs':{'long_name':'lead time','units':'days'}},
                "lon": {'dims':("lon",),
                          'data':lon,
                          'attrs':{'long_name':f'longitude','units':'degrees_east'}},
                "lat": {'dims':("lat",),
                          'data':lat,
                          'attrs':{'long_name':f'latitude','units':'degrees_north'}},
                }

        vardict = {f"{varname}_anom": {'dims':("lead_time","lat","lon"),
                                       'data':anom,
                                       'attrs':{'units':'degrees C' if varname=='T2m' else 'meters'}},
                   f"{varname}_spread": {'dims':("lead_time","lat","lon"),
                                       'data':spread,
                                       'attrs':{'units':'degrees C' if varname=='T2m' else 'meters'}},                    
                   f"{varname}_prob": {'dims':("lead_time","lat","lon"),
                                       'data':prob,
                                       'attrs':{'units':'percent probability above normal'}}
                        }

        if append_name is not None:
            varOut_name = varname+append_name
            save_ncds(vardict,coords,filename=os.path.join(save_to_path,f'{varOut_name}.{t_init:%Y%m%d}.nc'))
            del varOut_name
        else:
            save_ncds(vardict,coords,filename=os.path.join(save_to_path,f'{varname}.{t_init:%Y%m%d}.nc'))

    def plot_map(self,varname='T2m',t_init=None,lead_times=None,gridded=False,fullVariance=False,add_offset=None,\
                 categories='mean',save_to_path=None,nameconv='',prop={}):

        r"""
        Plots maps from PCs.

        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in the LIM

        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """

        #Set default properties
        default_prop={'cmap':None,'levels':None,'title':None,'figsize':(10,6),'dpi':120,
                      'drawcountries':True,'drawstates':True,'addtext':None}
        prop = add_prop(prop,default_prop)


        if t_init is None:
            t_init = max(self.RT_VARS['time'])

        ilt = np.array([self.lead_times.index(l) for l in lead_times])
        LT_lab = '-'.join([str(int(l/7)) for l in lead_times])
        fname_lab = '-'.join([f'{int(l):03}' for l in lead_times])
        varobj = self.use_vars[varname]['data']

        if gridded:
            FMAP = np.mean(self.F_recon[t_init][varname][ilt],axis=0)
            SMAP = np.mean(self.E_recon[t_init][varname][ilt],axis=0)

        else:
            if lead_times is not None:
                #check if lead_times is in self.lead_times
                try:
                    ilt = np.array([self.lead_times.index(l) for l in lead_times])
                    F_PC = np.mean(self.model_F[t_init][ilt],axis=0)
                    E_PC = np.mean(self.model_E[t_init][ilt],axis=0)
                except:
                    self.run_forecast(t_init.month,lead_times=lead_times)
                    F_PC = np.mean(self.model_F[t_init],axis=0)
                    E_PC = np.mean(self.model_E[t_init],axis=0)
            else:
                try:
                    F_PC = np.mean(self.model_F[t_init],axis=0)
                    E_PC = np.mean(self.model_E[t_init],axis=0)
                except:
                    self.run_forecast(t_init.month,lead_times=lead_times)
                    F_PC = np.mean(self.model_F[t_init],axis=0)
                    E_PC = np.mean(self.model_E[t_init],axis=0)

            FMAP,SMAP = self.pc_to_grid(F=F_PC,E=E_PC,limkey=self.RTLIMKEY,varname=varname,fullVariance=fullVariance,regrid=False)

        if add_offset is not None:
            ds = xr.open_dataset(add_offset)
            days = [int(f'{t_init+timedelta(days = lt):%j}') for lt in lead_times]
            newclim = np.mean([ds[varname].data[d-1] for d in days],axis=0)
            oldclim = np.mean([varobj.climo[d-1] for d in days],axis=0)

            diff = oldclim-newclim
            prepped = get_area_weighted(diff,varobj.lat)
            prepped = prepped / varobj.climo_stdev
            eof_lim = self.eof_trunc[t_init.month]
            eofobjs = self.eofobjs[t_init.month]
            pc = get_eofs(prepped,eof_in=eofobjs[varname].eof_dict['eof'][:eof_lim[varname]])
            diffrecon = eofobjs[varname].reconstruct(pc)[varname]

            FMAP = FMAP+diffrecon.squeeze()
            #print(f'RMSE: {np.mean(diffrecon**2)**.5}')

        if prop['cmap'] is None:
            prop['cmap'] = {0:'violet',22.5:'mediumblue',40:'lightskyblue',47.5:'w',52.5:'w',60:'gold',77.5:'firebrick',100:'violet'}
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(FMAP)),np.nanmax(abs(FMAP)))
        prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])
        prop['extend']='both'
        ax = varobj.plot_map(FMAP, prop = prop)
        ax.set_title(f'{varname} \nAnomaly',loc='left',fontweight='bold',fontsize=14)
        ax.set_title(f'Init: {t_init:%a %d %b %Y}\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)

        if prop['addtext'] is not None:
            ax.text( 0.04, 0.06, prop['addtext'], ha='left', va='bottom', transform=ax.transAxes,fontsize=9,zorder=99)

        #if save_to_path is None:
        #    plt.show()
        if not isinstance(save_to_path,str):
            print('WARNING: save_to_file must be a string indicating the path to save the figure to.')
        else:
            print('saving figure to: '+save_to_path+'/'+varname+'_lt'+fname_lab+'.png')
            plt.savefig(f'{save_to_path}/{varname}_lt{fname_lab}.png',bbox_inches='tight')
        plt.close()

        # make probabilistic forecast map

        if categories=='mean':
            bounds = [-np.inf*np.ones(len(FMAP))]+[np.zeros(len(FMAP))]+[np.inf*np.ones(len(FMAP))]
        else:
            ptiles = np.linspace(0,100,categories+1)[1:-1]
            newobj = copy.deepcopy(varobj)
            datebounds = (f'{t_init-timedelta(days=30):%m/%d}',f'{t_init+timedelta(days=30):%m/%d}')
            newobj.subset(datebounds=datebounds)
            climodata = newobj.running_mean
            pbounds = [np.percentile(climodata,p,axis=0) for p in ptiles]
            bounds = [-np.inf*np.ones(len(FMAP))]+pbounds+[np.inf*np.ones(len(FMAP))]

        cat_fcst = get_categorical_fcst((FMAP,),(SMAP,),bounds)[0]

        if categories in (2,'mean') and not np.all(np.isnan(cat_fcst[1])):
            cmap,levels = get_cmap_levels(prop['cmap'],np.arange(0,101,5))
            prop['cmap'] = cmap
            prop['levels'] = np.arange(0,101,5)
            prop['cbarticks'] = [0,10,20,30,40,60,70,80,90,100]
            prop['cbarticklabels']=['Below',90,80,70,60,60,70,80,90,'Above']
            prop['extend']='neither'
            prop['cbar_label']='%'
            ax = varobj.plot_map(cat_fcst[1]*100, prop = prop)
            #ax = plot_map(cat_fcst[1]*100,cmap = cmap, levels = levels, prop = prop)
            ax.set_title(f'{varname} \nProbability',loc='left',fontweight='bold',fontsize=14)
            ax.set_title(f'Init: {t_init:%a %d %b %Y}\n'+
                         f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                         loc='right',fontsize=14)

            if prop['addtext'] is not None:
                ax.text( 0.04, 0.06, prop['addtext'], ha='left', va='bottom', transform=ax.transAxes,fontsize=9,zorder=99)

            if save_to_path is None:
                plt.show()
            elif not isinstance(save_to_path,str):
                print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
            else:
                plt.savefig(f'{save_to_path}/{varname}-PROB_lt{fname_lab}.png',bbox_inches='tight')
            plt.close()

        if categories==3 and not np.all(np.isnan(cat_fcst[1])):

            cat0map = np.where(cat_fcst[0]>1/3,cat_fcst[0],0)*-1
            cat2map = np.where(cat_fcst[2]>1/3,cat_fcst[2],0)
            fcstmap = 100*(cat0map+cat2map)

            cmap,levels = get_cmap_levels({0:'violet',1/6:'mediumblue',1/3:'lightskyblue',1/3+.001:'w',2/3-.001:'w',2/3:'gold',5/6:'firebrick',1:'violet'},np.linspace(-100,100,256))
            prop['cmap'] = cmap
            prop['levels'] = [-100,-90,-80,-70,-60,-50,-40,-33,33,40,50,60,70,80,90,100]
            prop['cbarticks'] = [-100,-90,-80,-70,-60,-50,-40,-33,33,40,50,60,70,80,90,100]
            prop['cbarticklabels']=['Below',90,80,70,60,50,40,33,33,40,50,60,70,80,90,'Above']
            prop['extend']='neither'
            prop['cbar_label']='%'
            ax = varobj.plot_map(fcstmap, prop = prop)
            #ax = plot_map(cat_fcst[1]*100,cmap = cmap, levels = levels, prop = prop)
            ax.set_title(f'{varname} \nProbability',loc='left',fontweight='bold',fontsize=14)
            ax.set_title(f'Init: {t_init:%a %d %b %Y}\n'+
                         f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                         loc='right',fontsize=14)

            if prop['addtext'] is not None:
                ax.text( 0.04, 0.06, prop['addtext'], ha='left', va='bottom', transform=ax.transAxes,fontsize=9,zorder=99)

            if save_to_path is None:
                plt.show()
            elif not isinstance(save_to_path,str):
                print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
            else:
                plt.savefig(f'{save_to_path}/{varname}-PROB_lt{fname_lab}_terciles.png',bbox_inches='tight')
            plt.close()


    def plot_teleconnection(self,T_INIT=None,varname='H500',gridded=False,\
                            list_of_teleconnections = ['nao', 'ea', 'wp', 'epnp', 'pna', 'eawr', 'scand', 'tnh', 'poleur'],\
                            daysback=90,save_to_path=None,prop={}):

        r"""
        Plots teleconnection timeseries analysis, forecast, and spread.

        Parameters
        ----------
        list_of_teleconnections : list
            List of names (str) of teleconnections to plot.
        daysback : int
            Number of days prior to t_init to plot analysis.

        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """

        #Set default properties
        default_prop={'title':None,'dpi':120,'figsize':(12,6)}
        prop = add_prop(prop,default_prop)

        if T_INIT is None:
            T_INIT = max(self.model_F.keys())
        t1 = T_INIT-timedelta(days=daysback)

        eof_lim = self.eof_trunc[T_INIT.month]
        pci = np.cumsum([eof_lim[key] for key in eof_lim.keys()])
        vari = list(eof_lim.keys()).index(varname)
        varpci = [pci[vari]-eof_lim[varname],pci[vari]]
        varobj = self.use_vars[varname]['data']
        RTdata = self.RT_ANOM[varname]
        RTtime = self.RT_VARS['time']

        eofobj = {}
        for m in range(1,13):
            eofobj[m] = pickle.load( open( self.EOF_FILE_PREFIX+'+'.join(listify(varname))+f'_{m}.p', "rb" ) )

        anl = {t:y for t,y in zip(RTtime,RTdata) if t>=t1 and t<=T_INIT}
        #init = {t:self.F_recon[t][varname][0] for t in RTtime if t>=t1 and t<=T_INIT}

        # Last 7 forecasts (0 is present): valid time (datetime):
        # Reconstruct forecast PCs from EOFs

        if gridded:
            FCST = {dt0: {T_INIT+timedelta(days=i+1-dt0): f \
                          for i,f in enumerate(self.F_recon[T_INIT-timedelta(days=dt0)][varname])}\
                          for dt0 in range(7) if T_INIT-timedelta(days=dt0) in self.F_recon.keys()}
        else:
            FCST = {dt0: {T_INIT+timedelta(days=i+1-dt0): \
                eofobj[T_INIT.month].reconstruct(f[min(varpci):max(varpci)])[varname].squeeze() \
                for i,f in enumerate(self.model_F[T_INIT-timedelta(days=dt0)])} for dt0 in range(7) if T_INIT-timedelta(days=dt0) in self.model_F.keys()}

        # Read in teleconnection loading pattern file
        LOADING_PATTERNS = nc.Dataset(self.TELECONNECTION_PATTERN_NCFILE, 'r', format='NETCDF4')

        # Loading pattern grid
        plat = LOADING_PATTERNS['lat'][:]
        plon = LOADING_PATTERNS['lon'][:]

        def day_average(month_patterns,doy):
            date = dt.strptime(str(doy),'%j')
            mn2 = date.month
            mn1 = (mn2-2)%12+1
            mn3 = (mn2)%12+1
            mn2wt = 1
            mn3wt = date.day/31
            mn1wt = 1-mn3wt
            day_avg = np.nansum([w*month_patterns[m-1] for w,m in zip([mn1wt,mn2wt,mn3wt],[mn1,mn2,mn3])],axis=0)/2
            return day_avg

        self.TESTING ={}

        for NAME in list_of_teleconnections:

            LP = LOADING_PATTERNS[NAME][:].squeeze()
            # Interpolate loading pattern to the LIM variable grid
            LP_i = [interp2LIM(plat,plon,LP[mn],varobj) for mn in range(12)]
            # Interpolate monthly loading patterns to day of the year
            LP_interp = {dy:day_average(LP_i,dy) for dy in range(1,366)}

            # Projection of reanalysis data onto loading pattern
            PROJ = {t:np.dot(a,LP_interp[int(f'{t:%j}')%365+1])/len(a) for t,a in anl.items()}
            #PROJ_init = {t:np.dot(a,LP_interp[int(f'{t:%j}')%365+1])/len(a) for t,a in init.items()}
            # Last 7 forecasts (0 is present): valid time (datetime):
            # Projection of forecast data onto loading pattern
            PROJ_FCST = {dt0:{t:np.dot(f,LP_interp[int(f'{t:%j}')%365+1])/len(f) for t,f in FCST[dt0].items()} for dt0 in FCST.keys()}
            # Projection of spread onto loading pattern
            # sqrt( sum(spread^2 * pattern^2) )
#            PROJ_SPREAD = {t:np.sqrt(np.sum([(ff**2)*(lp**2) for ff,lp in zip(f,LP_interp[f'{t:%j}'])])) for t,f in SPREAD.items()}
            #PROJ_pc = {t:np.dot(varpcdict[t],LP_interp[t.month-1]) for t in varpcdict.keys()}

            PROJ_SPREAD = {t:(2.5*(t-T_INIT).days/len(FCST[0]))**0.5 for t,f in FCST[0].items()}
            PROJ_SPREAD[T_INIT] = 0
            import collections
            PROJ_SPREAD = collections.OrderedDict(sorted(PROJ_SPREAD.items()))

            fig = plt.figure(figsize=prop['figsize'],dpi=prop['dpi'])
            ax = plt.subplot()

            ax.plot(list(PROJ.keys()),list(PROJ.values()),color='blue',linewidth=2,linestyle='-',label='Analysis')
            #ax.plot(list(PROJ_init.keys()),list(PROJ_init.values()),color='blue',linewidth=1,linestyle='-',label='Initialization')

            ax.plot(list(PROJ_FCST[0].keys()),list(PROJ_FCST[0].values()),color='k',linewidth=1,\
                    marker='o',mfc='w')

            # Plot past 7 forecasts
            cmap = matplotlib.cm.get_cmap('plasma')
            for dt0 in list(FCST.keys())[:0:-1]:
                ax.plot(list(PROJ_FCST[dt0].keys()),list(PROJ_FCST[dt0].values()),color=cmap(dt0/7),linewidth=1,\
                        marker='o',mfc='w',ms=2,label=f'{T_INIT-timedelta(dt0):%d %b %Y}')
            ax.plot(list(PROJ_FCST[0].keys()),list(PROJ_FCST[0].values()),color='k',linewidth=1,\
                        marker='o',mfc='w',ms=5,label=f'{T_INIT:%d %b %Y}')

            # Plot fill between the forecast +/- 1 sigma spread
            anlandfcst = {**PROJ,**PROJ_FCST[0]}
            ax.fill_between(x=list(PROJ_SPREAD.keys()),\
                            y1=[anlandfcst[t]-PROJ_SPREAD[t] for t in PROJ_SPREAD.keys()],\
                            y2=[anlandfcst[t]+PROJ_SPREAD[t] for t in PROJ_SPREAD.keys()],\
                            color='0.8',zorder=0,label='1$\sigma$ Spread')

            ax.set_title(f'{LOADING_PATTERNS[NAME].long_name.upper()}',loc='left',fontsize=14,fontweight='bold')
            ax.set_title(f'Updated for {T_INIT:%a %d %b %Y}',loc='right',fontsize=12)

            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))

            ax.axis([T_INIT-timedelta(days=daysback),T_INIT+timedelta(days=int(1.5*len(FCST[0]))),-7,7])
            plt.plot([T_INIT-timedelta(days=daysback),T_INIT+timedelta(days=int(1.5*len(FCST[0])))],[0,0],c='k',lw=1.5,zorder=1)
            ax.grid(color='0.6')

            plt.legend(loc='upper left',fontsize=10)

            if save_to_path is None:
                plt.show()
            elif not isinstance(save_to_path,str):
                print('WARNING: save_to_path must be a string indicating the path of the directory to save the figures to.')
            else:
                plt.savefig(os.path.join(save_to_path,f'teleconnection_{NAME.upper()}.png'),bbox_inches='tight')
            plt.close()


    def plot_timelon(self,varname='colIrr',t_init=None,gridded=False,daysback=120,lat_bounds=(-7.5,7.5),add_offset=None,save_to_file=None,prop={}):

        r"""
        Plots meridionally averaged data with longitude on x-axis and time on y-axis

        Parameters
        ----------
        varname : str
            Name of variable (consistent with use_var dictionary keys) to plot. Default is colIrr
        t_init : datetime object
            Time of forecast initialization. Default is most recent time available.
        daysback : int
            Number of days prior to t_init to plot hovmoller
        lat_bounds : tuple or list
            Latitude boundaries to average data between. Default is (-7.5,7.5)
        save_to_file : str
            Name of file to save figure to. Default is None, which does not save the figure
        """

        #Set default properties
        default_prop={'cmap':'bwr','levels':None,'title':None,'dpi':120,'figsize':(6,10),'cbar_label':None}
        prop = add_prop(prop,default_prop)

        if t_init is None:
            t_init = max(self.RT_VARS['time'])

        t1 = t_init-timedelta(days=daysback)
        t2 = t_init
        endverif = len(self.RT_VARS['time'])-(max(self.RT_VARS['time'])-t2).days

        varobj = self.use_vars[varname]['data']

        lat_idx = np.where((varobj.latgrid>=min(lat_bounds)) & (varobj.latgrid<=max(lat_bounds)))[0]

#        ytime = [t for t in varobj.time if t>=t1 and t<=t2]
#        varplot = np.array([np.mean(varobj.regrid(x)[min(lat_idx):max(lat_idx)+1,:],axis=0) \
#               for t,x in zip(varobj.time,varobj.running_mean) if t in ytime])
        ytime = [t2+timedelta(days=i+1-daysback) for i in range(daysback)]
        varplot = [np.nanmean(varobj.regrid(x)[min(lat_idx):max(lat_idx)+1,:],axis=0) \
               for t,x in zip(ytime,self.RT_ANOM[varname][max([0,endverif-daysback]):endverif])]
        varplot = np.array([np.ones(varobj.longrid.shape[1])*np.nan]*(len(ytime)-len(varplot)) + varplot)
        xlon = varobj.longrid[0,:]

        eof_lim = self.eof_trunc[self.RTLIMKEY]
        varpci = get_varpci(eof_lim,varname)
        eofobj = self.eofobjs[self.RTLIMKEY][varname]

        if gridded:
            F = list(map(varobj.regrid,self.F_recon[t2][varname]))
        else:
            F = self.pc_to_grid(F=self.model_F[t2],limkey=self.RTLIMKEY,varname=varname,fullVariance=fullVariance,regrid=True)

        if add_offset is not None:
            print('getting offset')
            ds = xr.open_dataset(add_offset)
            days = [int(f'{t_init+timedelta(days = lt):%j}') for lt in range(len(F))]
            newclim = np.mean([ds[varname].data[d-1] for d in days],axis=0)
            oldclim = np.mean([varobj.climo[d-1] for d in days],axis=0)

            diff = oldclim-newclim
            prepped = get_area_weighted(diff,varobj.lat)
            prepped = prepped / varobj.climo_stdev
            eof_lim = self.eof_trunc[t_init.month]
            eofobjs = self.eofobjs[t_init.month]
            pc = get_eofs(prepped,eof_in=eofobjs[varname].eof_dict['eof'][:eof_lim[varname]])
            diffrecon = eofobjs[varname].reconstruct(pc)[varname]

            F = F+varobj.regrid(diffrecon.squeeze())
            print(f'RMSE: {np.mean(diffrecon**2)**.5}')

        Fvar = np.array([np.mean(x[min(lat_idx):max(lat_idx)+1,:],axis=0) for x in F])

        FORECAST = {}
        FORECAST['dates'] = [t2+timedelta(days=i+1) for i in range(len(Fvar))]
        FORECAST['var'] = Fvar

        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(varplot)),np.nanmax(abs(varplot)))
        cmap,clevs = get_cmap_levels(prop['cmap'],prop['levels'])

        fig = plt.figure(figsize=prop['figsize'],dpi=prop['dpi'])
        ax = plt.subplot()
        if prop['title'] is None:
            prop['title'] = varname
        ax.set_title(f'{prop["title"]}',fontweight='bold',fontsize=16,loc='left')
        ax.set_title(f'{lat2str(min(lat_bounds))} – {lat2str(max(lat_bounds))}',fontsize=16,loc='right')
        _=timelonplot(ax,xlon,ytime,varplot,FORECAST=FORECAST,\
                      cmap=cmap,levels=clevs,cbar_label=prop['cbar_label'])

        if save_to_file is None:
            plt.show()
        elif not isinstance(save_to_file,str):
            print('WARNING: save_to_file must be a string indicating the path and name of the file to save the figure to.')
        else:
            plt.savefig(save_to_file,bbox_inches='tight')
            plt.close()


    def plot_timelat(self,varname,t_init=None,daysback=120,lon_bounds=(0,360),save_to_file=None,prop={}):

        r"""
        Plots zonally averaged data with latitude on x-axis and time on y-axis

        Parameters
        ----------
        varname : str
            Name of variable (consistent with use_var dictionary keys) to plot
        t_init : datetime object
            Time of forecast initialization
        lon_bounds : tuple or list
            Longitude boundaries to average data between.
        """

        #Set default properties
        default_prop={'cmap':'bwr','levels':None,'title':None,'dpi':150}
        prop = add_prop(prop,default_prop)


    def plot_mjo(self,t_init,daysback=60,prop={},save_to_file=None):

        r"""
        Plots MJO analysis, forecast and spread in 2-dimensional RMM phase space

        Parameters
        ----------
        t_init : datetime object
            Time of forecast initialization. Default is most recent time available.
        daysback : int
            Number of days prior to t_init to plot MJO analysis
        save_to_file : str
            Name of file to save figure to. Default is None, which does not save the figure
        """

        #Set default properties
        default_prop={'title':None,'dpi':150,'figsize':(10,10)}
        prop = add_prop(prop,default_prop)

        plot_type = 'MJO'

        # path to data file
        URL ='http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'

        # dates to plot (t1,t2) -- dt(yyyy,m,d)
        t2 = t_init
        t1 = t2-timedelta(days=40)
        date_range = (t1,t2)

        # LOAD DATA FUNCTION

        def get_data(URL,date_range=None):
            headers={'User-Agent':'whatever'}
            s=requests.get(URL, headers=headers).text
            data = np.loadtxt(StringIO(s),unpack=True,skiprows=2,usecols=(0,1,2,3,4))

            DATA={}
            DATA['x']=data[-2] # x-component of data here
            DATA['y']=data[-1] # y-component of data here
            DATA['dates']=[dt(int(y),int(m),int(d)) for y,m,d in zip(data[0],data[1],data[2])] # dates here
            DATA = pandas.DataFrame.from_dict(DATA)

            if date_range is None:
                date_range = (min(DATA['dates']),max(DATA['dates']))
            subDATA = DATA.loc[(DATA['dates']>=min(date_range)) & \
                                       (DATA['dates']<=max(date_range))]
            return subDATA.reset_index()

        # define colors for monthly segments
        cmap=mlib.cm.get_cmap('gnuplot2_r')

        lenDATA = len(DATA['dates'])
        if FORECAST is None:
            colorlist=cmap(np.linspace(0.3, 0.9, int(lenDATA/30)+2))
        else:
            lenFORECAST = len(FORECAST['dates'])
            colorlist=cmap(np.linspace(0.3, 0.9, int((lenDATA+lenFORECAST)/30)+2))

        # finally make the plot
        plt.figure(**kwargs)
        ax=plt.subplot()
        imonth=-1
        for i in range(lenDATA):
            txtamp=(DATA['x'][i]**2+DATA['y'][i]**2)**0.5+0.16
            txtang=math.atan2(DATA['y'][i],DATA['x'][i])
            # label every 5 days, 5 through 25
            if  DATA['dates'][i].day in np.arange(5,31,5):
                plt.text(txtamp*np.cos(txtang),txtamp*np.sin(txtang),str(DATA['dates'][i].day),
                         ha='center', va='center', color=colorlist[imonth], fontsize=12, zorder=1)
            # color-code and label monthly segments
            if  (DATA['dates'][i].day == 1) or (i == 0):
                imonth+=1
                txtamp=(DATA['x'][i]**2+DATA['y'][i]**2)**0.5+0.3
                plt.text(-3.9,3.15-imonth*.35,DATA['dates'][i].strftime('%b'),
                         ha='left', va='center', color=colorlist[imonth], fontsize=15, fontweight='bold',zorder=99)
    #            plt.text(txtamp*np.cos(txtang),txtamp*np.sin(txtang),DATA['dates'][i].strftime('%b'),
    #                     ha='center', va='center', color=colorlist[imonth], fontsize=15, zorder=99)
            ax.plot(DATA['x'][i:i+2],DATA['y'][i:i+2],color=colorlist[imonth],linewidth=2,zorder=0)
            ax.scatter(DATA['x'][i],DATA['y'][i],color='k',s=10)

        # ADD FORECAST
        if FORECAST is not None:
            PLOTS = {}

            if FORECAST['spread'] is not None:
                cone_dict = generate_cone(FORECAST)
                cone_2d = ndimage.gaussian_filter(cone_dict['cone'],sigma=0.5,order=0)
                ax.contourf(cone_dict['lon2d'],cone_dict['lat2d'],cone_2d,
                            [0.9,1.1],colors=['0.7','0.7'],zorder=0)
                tmp = plt.Rectangle((0,0),1,1,fc = '0.7')
                PLOTS['68% Confidence']=tmp

            tmp, = ax.plot(FORECAST['x'],FORECAST['y'],color='k',linewidth=4.5,zorder=0)
            PLOTS['Forecast']=tmp

            for i in range(lenFORECAST):
                txtamp=(FORECAST['x'][i]**2+FORECAST['y'][i]**2)**0.5+0.16
                txtang=math.atan2(FORECAST['y'][i],FORECAST['x'][i])
                # label every 5 days, 5 through 25
                if  FORECAST['dates'][i].day in np.arange(5,31,5):
                    plt.text(txtamp*np.cos(txtang),txtamp*np.sin(txtang),str(FORECAST['dates'][i].day),
                             ha='center', va='center', color=colorlist[imonth], fontsize=12, zorder=1)
                # color-code and label monthly segments
                if  (FORECAST['dates'][i].day == 1):
                    imonth+=1
                    plt.text(-3.9,3.15-imonth*.35,FORECAST['dates'][i].strftime('%b'),
                         ha='left', va='center', color=colorlist[imonth], fontsize=15, fontweight='bold',zorder=99)
                    txtamp=(FORECAST['x'][i]**2+FORECAST['y'][i]**2)**0.5+0.3
    #                plt.text(txtamp*np.cos(txtang),txtamp*np.sin(txtang),FORECAST['dates'][i].strftime('%b'),
    #                         ha='center', va='center', color=colorlist[imonth], fontsize=15, zorder=99)
                ax.plot(FORECAST['x'][i:i+2],FORECAST['y'][i:i+2],color=colorlist[imonth],linewidth=2,zorder=0)
                ax.scatter(FORECAST['x'][i],FORECAST['y'][i],color='k',marker='o',s=10)

            print(PLOTS)
            ax.legend(list(PLOTS.values()),list(PLOTS.keys()),loc='upper left',fontsize=12)

            plt.suptitle(plot_type+' phase space',fontname='Times New Roman',fontsize=28)
            ax.text(0,4.05,f'Forecast for {min(FORECAST["dates"]):%d-%b-%Y} to {max(FORECAST["dates"]):%d-%b-%Y}\n',
                    va='center',ha='center',fontname='Times New Roman',fontsize=22)

        else:
            plt.suptitle(plot_type+' phase space',fontname='Times New Roman',fontsize=28)
            ax.text(0,4.05,'for '+min(DATA['dates']).strftime('%d-%b-%Y')+' to '+\
                    max(DATA['dates']).strftime('%d-%b-%Y')+'\n',
                    va='center',ha='center',fontname='Times New Roman',fontsize=22)

        # call the phase space function
        phase_space(ax=ax,plot_type=plot_type,axlim=4)
        plt.savefig(filename,bbox_inches='tight')


    def plot_verif(self,varname='T2m',t_init=None,lead_times=None,Fmap=None,Emap=None,
                   add_offset=None,prob_thresh=50,regMask=None,save_to_path=None,prop={}):

        r"""
        Plots verifying anomalies.

        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in RT_VARS

        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """

        #Set default properties
        default_prop={'cmap':None,'levels':None,'title':None,'figsize':(10,6),'dpi':120,
                      'drawcountries':True,'drawstates':True}
        prop = add_prop(prop,default_prop)

        lead_times = listify(lead_times)
        if lead_times is None:
            lead_times = self.lead_times
        if t_init is None:
            t_init = max(self.RT_VARS['time'])-timedelta(days=max(lead_times))
        t_verif = np.array([t_init + timedelta(days=i) for i in lead_times])

        itv = np.in1d(self.RT_VARS['time'],t_verif).nonzero()[0]
        #check if all t_verif in RT_VARS, otherwise exit with error
        if len(itv)<len(listify(t_verif)):
            print(f'{t_verif} not in RT_VARS')
            sys.exit(1)

        varobj = self.use_vars[varname]['data']

        #check if lead_times is in self.lead_times
        if Fmap is None and Emap is None:
            try:
                ilt = np.array([self.lead_times.index(l) for l in lead_times])
                F_PC = np.mean(self.model_F[t_init][ilt],axis=0)
                E_PC = np.mean(self.model_E[t_init][ilt],axis=0)
                outF,outE = self.pc_to_grid(F=F_PC,E=E_PC,regrid=False)
                FMAP,EMAP = outF[varname],outE[varname]
            except:
                print(f'lead times {lead_times} and/or t_init {t_init} not available in model_F.')
        else:
            if len(Fmap.shape)>1:
                FMAP = Fmap[varobj.domain]
            else:
                FMAP = Fmap
            if len(Emap.shape)>1:
                EMAP = Emap[varobj.domain]
            else:
                EMAP = Emap

        fname_lab = '-'.join([f'{int(l):03}' for l in lead_times])

        VMAP = np.mean(self.RT_ANOM[varname][itv],axis=0)

        if add_offset is not None:
            print('getting offset')
            ds = xr.open_dataset(add_offset)
            days = [int(f'{t_init+timedelta(days = lt):%j}') for lt in lead_times]
            newclim = np.mean([ds[varname].data[d-1] for d in days],axis=0)
            oldclim = np.mean([varobj.climo[d-1] for d in days],axis=0)

            diff = oldclim-newclim
            prepped = get_area_weighted(diff,varobj.lat)
            prepped = prepped / varobj.climo_stdev
            eof_lim = self.eof_trunc[t_init.month]
            eofobjs = self.eofobjs[t_init.month]
            pc = get_eofs(prepped,eof_in=eofobjs[varname].eof_dict['eof'][:eof_lim[varname]])
            diffrecon = eofobjs[varname].reconstruct(pc)[varname]
            FMAP = FMAP+diffrecon.squeeze()
            print(f'RMSE: {np.mean(diffrecon**2)**.5}')

            pc = get_eofs(prepped,eof_in=eofobjs[varname].eof_dict['eof'][:eof_lim[varname]*2])
            diffrecon = eofobjs[varname].reconstruct(pc)[varname]
            VMAP = VMAP+diffrecon.squeeze()
            print(f'RMSE: {np.mean(diffrecon**2)**.5}')


        if prop['cmap'] is None:
            prop['cmap'] = {0:'violet',22.5:'mediumblue',40:'lightskyblue',47.5:'w',52.5:'w',60:'gold',77.5:'firebrick',100:'violet'}
        if prop['levels'] is None:
            prop['levels'] = (-np.nanmax(abs(VMAP)),np.nanmax(abs(VMAP)))
        prop['cmap'],prop['levels'] = get_cmap_levels(prop['cmap'],prop['levels'])
        prop['extend']='both'
        ax = varobj.plot_map(VMAP, prop = prop)
        ax.set_title(f'{varname} \nAnomaly',loc='left',fontweight='bold',fontsize=14)
        ax.set_title(f'Verification\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)

        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_file must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}_lt{fname_lab}_obs.png',bbox_inches='tight')
        plt.close()

        bounds = [-np.inf*np.ones(len(FMAP)),np.zeros(len(FMAP)),np.inf*np.ones(len(FMAP))]
        cat_fcst = get_categorical_fcst((FMAP,),(EMAP,),bounds)[0]
        cat_obs = get_categorical_obs((VMAP,),bounds)[0]

        # make categorical verification map
        cmap,levels = get_cmap_levels(['deepskyblue','coral'],[0,1])
        prop['cmap'] = cmap
        prop['levels'] = [0,.5,1]
        prop['cbarticklabels']=['Below','','Above']
        prop['cbar_label']=None
        prop['extend']='both'
        ax = varobj.plot_map(cat_obs[1], prop = prop)
        ax.set_title(f'{varname} \nCategorical',loc='left',fontweight='bold',fontsize=14)
        ax.set_title(f'Verification\n'+
                     f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                     loc='right',fontsize=14)

        if save_to_path is None:
            plt.show()
        elif not isinstance(save_to_path,str):
            print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
        else:
            plt.savefig(f'{save_to_path}/{varname}-CAT_lt{fname_lab}_obs.png',bbox_inches='tight')
        plt.close()

        # make hit/miss map
        if regMask is not None:
            prop['regMask'] = regMask
            countries = regionmask.defined_regions.natural_earth.countries_110
            regIndex = [i for i in countries.regions if countries.regions[i].name==regMask][0]
            labels = countries.mask(varobj.longrid,varobj.latgrid)
            regMask = np.where(labels.data[varobj.domain]==regIndex,True,False)
        else:
            regMask = np.ones(varobj.lon.shape).astype(bool)

        cmap,levels = get_cmap_levels(['lightpink','mediumseagreen'],[-1,1])
        prop['cmap'] = cmap
        prop['levels'] = [-1,0,1]
        prop['cbarticklabels']=['Miss','','Hit']
        prop['cbar_label']=None
        prop['extend']='both'
        hitmiss=np.sign((2*cat_obs[1]-1)*(2*cat_fcst[1]-1))

        for pthresh in listify(prob_thresh):

            validwhere = (abs(cat_fcst[1]-.5)<(pthresh/100-.5))

            latwt = np.cos(np.radians(varobj.lat))
            fcst = np.array(cat_fcst).T
            obs = np.array(cat_obs).T

            HSS = get_heidke(fcst[regMask],obs[regMask],weights=latwt[regMask],\
                                        categorical=True)
            HSS_thresh = get_heidke(fcst[~validwhere & regMask],\
                                    obs[~validwhere & regMask],\
                                        weights=latwt[~validwhere & regMask],\
                                        categorical=True)

            RPSS = get_rpss(fcst,obs,weights=latwt,\
                                       categorical=False)
            RPSS_thresh = get_rpss(fcst[~validwhere & regMask],\
                                   obs[~validwhere & regMask],\
                                       weights=latwt[~validwhere & regMask],\
                                       categorical=False)

            mask = (pthresh/100-0.5)-(abs(cat_fcst[1]-.5))
            ax = varobj.plot_map(hitmiss, mask=mask, prop = prop)
            ax.set_title(f'{varname} \nHit/Miss >{pthresh}%',loc='left',fontweight='bold',fontsize=14)
            ax.set_title(f'Verification\n'+
                         f'Valid: {t_init+timedelta(days=min(lead_times)-6):%d %b} – {t_init+timedelta(days=max(lead_times)):%d %b}',
                         loc='right',fontsize=14)
            ax.text( 0.03, 0.12, f'Heidke (all) = {HSS:.3f} \nHeidke (>{pthresh}%) = {HSS_thresh:.3f}'+\
                    f'\nRPSS (all) = {RPSS:.3f} \nRPSS (>{pthresh}%) = {RPSS_thresh:.3f}',
                    ha='left', va='center', transform=ax.transAxes,fontsize=12,zorder=99)

            if save_to_path is None:
                plt.show()
            elif not isinstance(save_to_path,str):
                print('WARNING: save_to_path must be a string indicating the path to save the figure to.')
            else:
                plt.savefig(f'{save_to_path}/{varname}-CAT_lt{fname_lab}_hitmiss_{pthresh}.png',bbox_inches='tight')
            plt.close()

        return {'HSS_all':HSS,f'HSS_{pthresh}':HSS_thresh,'RPSS_all':RPSS,f'RPSS_{pthresh}':RPSS_thresh}


    # def calc_verif(self,t_init,lead_times,F,E,varname='T2m',
    #                prob_thresh=50,regMask=None):

    #     r"""
    #     Plots verifying anomalies.

    #     Parameters
    #     ----------
    #     varname : str
    #         Must be a variable name in the list of keys used in RT_VARS

    #     """

    #     if lead_times is None:
    #         lead_times = self.lead_times
    #     lead_times = listify(lead_times)

    #     t_verif = np.array([t_init + timedelta(days=int(i)) for i in lead_times])

    #     itv = np.in1d(self.RT_VARS['time'],t_verif).nonzero()[0]
    #     #check if all t_verif in RT_VARS, otherwise exit with error
    #     if len(itv)<len(listify(t_verif)):
    #         print(f'{t_verif} not in RT_VARS')
    #         sys.exit(1)

    #     varobj = self.use_vars[varname]['data']

    #     VMAP = self.RT_ANOM[varname][itv]

    #     # probabilistic forecast and categorical observations
    #     bounds = [-np.inf*np.ones(varobj.lon.shape),np.zeros(varobj.lon.shape),np.inf*np.ones(varobj.lon.shape)]
    #     cat_fcst = get_categorical_fcst(F,E,bounds)
    #     cat_obs = get_categorical_obs(VMAP,bounds)

    #     # geographical mask
    #     if regMask is not None:
    #         countries = regionmask.defined_regions.natural_earth.countries_110
    #         regIndex = [i for i in countries.regions if countries.regions[i].name==regMask][0]
    #         labels = countries.mask(varobj.longrid,varobj.latgrid)
    #         regMask = np.where(labels.data[varobj.domain]==regIndex,True,False)
    #     else:
    #         regMask = np.ones(varobj.lon.shape).astype(bool)


    #     # loop over all lead times
    #     all_skill = []
    #     for F,O in zip(cat_fcst,cat_obs):

    #         # for prob_thresh cutoff
    #         validwhere = (abs(F[1]-.5)<(prob_thresh/100-.5))

    #         latwt = np.cos(np.radians(varobj.lat))
    #         HSS_fcst = np.round(np.array(F).T) #ROUNDED TO 0 and 1
    #         HSS_obs = np.array(O).T
    #         HSS_thresh = get_heidke(HSS_fcst[~validwhere & regMask],\
    #                                 HSS_obs[~validwhere & regMask],\
    #                                     weights=latwt[~validwhere & regMask])
    #         all_skill.append(HSS_thresh)

    #     return all_skill


    def calc_verif(self,varname='T2m',t_init=None,lead_times=None,Fmap=None,Emap=None,limkey=None,
                   prob_thresh=50,regMask=None,save_to_path=None,prop={}):

        r"""
        Plots verifying anomalies.

        Parameters
        ----------
        varname : str
            Must be a variable name in the list of keys used in RT_VARS

        Other Parameters
        ----------------
        prop : dict
            Customization properties for plotting
        """

        #Set default properties
        default_prop={'cmap':None,'levels':None,'title':None,'figsize':(10,6),'dpi':200,
                      'drawcountries':True,'drawstates':True}
        prop = add_prop(prop,default_prop)

        lead_times = listify(lead_times)
        if lead_times is None:
            lead_times = self.lead_times
        if t_init is None:
            t_init = max(self.RT_VARS['time'])-timedelta(days=max(lead_times))
        t_verif = np.array([t_init + timedelta(days=i) for i in lead_times])

        itv = np.in1d(self.RT_VARS['time'],t_verif).nonzero()[0]
        #check if all t_verif in RT_VARS, otherwise exit with error
        if len(itv)<len(listify(t_verif)):
            print(f'{t_verif} not in RT_VARS')
            sys.exit(1)

        varobj = self.use_vars[varname]['data']

        #check if lead_times is in self.lead_times
        if Fmap is None and Emap is None:
            try:
                ilt = np.array([self.lead_times.index(l) for l in lead_times])
                F_PC = np.mean(self.model_F[t_init][ilt],axis=0)
                E_PC = np.mean(self.model_E[t_init][ilt],axis=0)
                outF,outE = self.pc_to_grid(F=F_PC,E=E_PC,limkey=limkey,regrid=False)
                FMAP,EMAP = outF[varname],outE[varname]
            except:
                print(f'lead times {lead_times} and/or t_init {t_init} not available in model_F.')
        else:
            if len(Fmap.shape)>1:
                FMAP = Fmap[varobj.domain]
            else:
                FMAP = Fmap
            if len(Emap.shape)>1:
                EMAP = Emap[varobj.domain]
            else:
                EMAP = Emap


        VMAP = np.mean(self.RT_ANOM[varname][itv],axis=0)

        bounds = [-np.inf*np.ones(len(FMAP)),np.zeros(len(FMAP)),np.inf*np.ones(len(FMAP))]
        cat_fcst = get_categorical_fcst((FMAP,),(EMAP,),bounds)[0]
        cat_obs = get_categorical_obs((VMAP,),bounds)[0]

        # make hit/miss map
        if regMask is not None:
            prop['regMask'] = regMask
            countries = regionmask.defined_regions.natural_earth.countries_110
            regIndex = [i for i in countries.regions if countries.regions[i].name==regMask][0]
            labels = countries.mask(varobj.longrid,varobj.latgrid)
            regMask = np.where(labels.data[varobj.domain]==regIndex,True,False)
        else:
            regMask = np.ones(varobj.lon.shape).astype(bool)

        for pthresh in listify(prob_thresh):

            validwhere = (abs(cat_fcst[1]-.5)<(pthresh/100-.5))

            latwt = np.cos(np.radians(varobj.lat))
            fcst = np.array(cat_fcst).T
            obs = np.array(cat_obs).T
            HSS = get_heidke(fcst[regMask],obs[regMask],weights=latwt[regMask],\
                                        categorical=True)
            HSS_thresh = get_heidke(fcst[~validwhere & regMask],\
                                    obs[~validwhere & regMask],\
                                        weights=latwt[~validwhere & regMask],\
                                        categorical=True)

            RPSS = get_rpss(fcst,obs,weights=latwt,\
                                       categorical=False)
            RPSS_thresh = get_rpss(fcst[~validwhere & regMask],\
                                   obs[~validwhere & regMask],\
                                       weights=latwt[~validwhere & regMask],\
                                       categorical=False)

        return {'HSS_all':HSS,f'HSS_{pthresh}':HSS_thresh,'RPSS_all':RPSS,f'RPSS_{pthresh}':RPSS_thresh}
