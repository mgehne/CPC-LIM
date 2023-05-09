r"""

Edited: J.R. Albers 10.4.2022
This functions contains a collection of tools for manipulating LIM variables and forecast objects, including various plotting and skill calculations


"""

####################################################################################
# IMPORT PACKAGES
####################################################################################

import sys
import numpy as np
import netCDF4 as nc
from scipy import signal as sig, linalg, stats
from scipy import fft
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter as gfilt
from scipy.optimize import leastsq
from scipy import fft
from datetime import datetime as dt,timedelta
import matplotlib as mlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ColorConverter
from mpl_toolkits.basemap import Basemap,shiftgrid,addcyclic
import xarray as xr
import copy

#%%

####################################################################################
# MAIN CODE BODY
####################################################################################

def get_coarsegrain(data,lats,lons,binsize):
    new_lats = np.arange(90,-91,-binsize)
    new_lons = np.arange(0,360,binsize)
    new_lons,new_lats = np.meshgrid(new_lons,new_lats)
    new_lons_flat = new_lons.flatten()
    new_lats_flat = new_lats.flatten()

    def get_area_avg(lat0,lon0):
        area = np.where((abs(lats-lat0)<=binsize*.5) & (abs(lons-lon0)<=binsize*.5))
        area_avg = np.sum(data[area]*np.cos(lats[area]*np.pi/180)) / np.sum(np.cos(lats[area]*np.pi/180))
        return area_avg

    seq = map(get_area_avg,new_lats_flat,new_lons_flat)
    output = np.asarray(list(seq)).reshape(new_lats.shape)
    return output,new_lats,new_lons

def listify(x):
    if isinstance(x,(tuple,list)):
        return [i for i in x]
    else:
        return [x]

def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

def eof_reconstruct(eofobj,pcs,num_eofs=None):
    if not isinstance(pcs,np.ndarray):
        pcs = np.array(pcs).squeeze()
    varobj = eofobj.varobjs[0]
    lat = varobj.ds['lat']
    lon = varobj.ds['lon']
    e = eofobj.eof_dict['eof']
    if num_eofs is None:
        num_eofs = min(e.shape[0],pcs.shape[1])
    recon = np.dot(pcs[:,:num_eofs],e[:num_eofs, :])
    recon_maps = np.array(recon).reshape([len(recon),len(lat),len(lon)],order='C')
    return recon_maps

def get_climo(data,time,yearbounds):
    #print('--> Starting to calculate climatology')
    timer_start = dt.now()

    doy = np.array([int(dt.strftime(i,'%j'))-1 for i in time])
    year = np.array([t.year for t in time])
    if yearbounds is None:
    	yearbounds = (min(year),max(year))
    doy = doy[np.where((year>=min(yearbounds)) & (year<=max(yearbounds)))]
    data = data[np.where((year>=min(yearbounds)) & (year<=max(yearbounds)))]

    def fit_harm(d):
        t=doy; num=4
        guess_mean = np.mean(d)
        guess_amp = np.std(d)
        optimize_func = lambda x: np.sum(x[0]+[x[i-1]*np.sin((i*np.pi/365)*t+x[i]) for i in range(2,num*2+1,2)],axis=0) - d
        x = leastsq(optimize_func, [guess_mean]+[guess_amp,0]*num)[0]
        d_fit = np.sum(x[0]+[x[i-1]*np.sin((i*np.pi/365)*t+x[i]) for i in range(2,num*2+1,2)],axis=0)
        return d_fit

#    climo = np.apply_along_axis(fit_harm,0,data)
    
#    climo=[np.nanmean(data[np.where(doy%365==i)],axis=0) for i in range(365)]
#    climo = gfilt(3*climo,[15]+[0]*len(data.shape[1:]))[365:2*365]
    #climo = data[0:366,:,:].values
    climo = np.array([np.nanmean(data[np.where(doy%365==i)],axis=0) for i in range(365)])
    cfft = fft.rfft(climo,n=365,axis=0)
    cfft[4:,:] = 0
    climo = fft.irfft(cfft,n=365,axis=0)

    print('--> Completed calculating climatology (%.1f seconds)' \
          % (dt.now()-timer_start).total_seconds())
    return climo

def get_anomaly(data,time,climo):
    #print('--> Starting to calculate anomaly')
    timer_start = dt.now()
    doy = np.array([int(dt.strftime(i,'%j'))-1 for i in time])
    anomaly = np.zeros(data.shape)
    for i,j in enumerate(climo):
        try:
            anomaly[np.where(doy%365==i)] = data[np.where(doy%365==i)] - j
        except:
            pass
    print('--> Completed calculating anomaly (%.1f seconds)' \
          % (dt.now()-timer_start).total_seconds())
    return anomaly

def get_running_mean(data,time_window,verbose=False):
    r"""
    Calculates running mean of data

    Parameters
    ----------
    time_window : int
        Length of the window for averaging, in days.

    Returns
    -------
    ndarray-like
        Data smoothed using the running mean.
    """

    timer_start = dt.now()
    filt = [1./float(time_window)]*int(time_window)
    running_mean = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, arr=data)
    running_mean = np.append(np.ones([len(data)-len(running_mean),*data.shape[1:]])*np.nan,running_mean,axis=0)
    if verbose: print(f'--> Completed calculating running mean ({(dt.now()-timer_start).total_seconds():.1f} seconds)')
    return running_mean

#def get_incirc(lat0,lon0,lats,lons,rad):
#    dist = unitdist(lat0,lon0,lats,lons)
#    incirc = np.where(dist<=rad)
#    return incirc
#
#def get_area_mean(data,lats,lons,rad):
#    datashape = data.shape
#    latwts = np.cos(np.radians(lats))
#

def get_detrended(data):
    dummydata = data.copy()
    ind = np.where(~np.isnan(data))[0]
    first,last = ind[0],ind[-1]
    dummydata[:first] = dummydata[first]
    dummydata[last+1:] = dummydata[last]
    detrended = sig.detrend(dummydata,axis=0)
    detrended[np.isnan(data)]=np.nan
    return detrended

def get_area_weighted(data,lats,cospow=.5):
    # reshape lats array to be consistent with data
    dshape = data.shape
    lshape = lats.shape
    newlshape = (1,)*(len(dshape)-len(lshape)-1)+lats.shape+(1,)*(2-len(lshape))
    lats = lats.reshape(newlshape)
    latwts = np.cos(np.radians(lats))**cospow
    latwts[np.where(lats==90)]=0
    area_weighted = data*latwts.T
    return area_weighted

def get_eofs(data, max_eofs=None, resample=None, eof_in=None):
    r"""
    Calculate spatial EOFs on the data retaining a specified number of
    modes.

    Parameters
    ----------
    num_eofs: int
        How many modes to retain from the EOF decomposition. Ignored if
        input_eofs is specified.
    eof_in: ndarray, optional
        A set of EOFs to project the data onto. First dimension should
        match the length of the data feature dimension. Overrides
        num_eofs if provided.

    Returns
    -------
        ndarray-like
        Principal components from data projected onto EOFs. Will have shape
        of (sampling dim x num EOFs).
    """

    # detrend data and weight by area of points
    #prepped = get_detrended(data)
    prepped = data
    prepped[np.isnan(prepped)]=0
    prepped = np.ma.getdata(prepped)
    # reshape to 2d array
    pshape = prepped.shape
    (nt,ns) = (pshape[0],np.product(pshape[1:]))
    prepped = prepped.reshape((nt,ns),order='F')

    if max_eofs is None:
        max_eofs = ns

    if eof_in is None:
        print('--> Starting to calculate EOFs (takes a minute for large datasets)')
        timer_start = dt.now()
        U, S, V = linalg.svd(prepped, full_matrices=False)
        U = U[:,:max_eofs]
        S = S[:max_eofs]
        V = V[:max_eofs]
        out_eofs = V
        out_pcs = np.dot(U,np.diag(S)) #pcs[:, :max_eofs]
        out_var = data[:].var(ddof=1, axis=0)
        print('--> Completed calculating EOFs (%.1f seconds)' \
              % (dt.now()-timer_start).total_seconds())
    else:
        if eof_in.shape[1] != ns:
            print('Input EOFs feature dimension (length={}) does '
                         'not match data feature dimension (length={})'
                         ''.format(eof_in.shape[1], prepped.shape[1]))
            raise ValueError('Feature dimension mismatch for input EOFs')
        else:
            max_eofs = np.min([max_eofs,eof_in.shape[0]])
            #print('Projecting data into leading {:d} EOFs'.format(max_eofs))
            out_pcs = np.matmul(eof_in[:max_eofs],prepped.T).T
            return out_pcs

    eig_vals = (S ** 2) / nt
    total_var = out_var.sum()
    var_expl_by_mode = eig_vals / total_var

    eof_dict = {}
    eof_dict['eof'] = out_eofs
    eof_dict['total_var'] = total_var
    eof_dict['var_expl_by_eof'] = var_expl_by_mode
    eof_dict['pc'] = out_pcs

    return eof_dict

#%%

def calc_lac(fcst,obs):
    # Calculate means of data
    f_mean = np.mean(fcst,axis=0)
    o_mean = np.mean(obs,axis=0)
    f_anom = fcst - f_mean
    o_anom = obs - o_mean

    # Calculate covariance between time series at each gridpoint
    cov = (f_anom * o_anom).sum(axis=0)

    # Calculate standardization terms
    f_var = (f_anom**2).sum(axis=0)
    o_var = (o_anom**2).sum(axis=0)
    f_std = np.sqrt(f_var)
    o_std = np.sqrt(o_var)

    std = f_std * o_std
    lac = cov / std
    return lac

def calc_lac_ifcst(fcst,obs,ifcst):
    F_list = [np.array(fcst).T[i][j] for i,j in enumerate(ifcsts)]
    O_list = [np.array(obs).T[i][j] for i,j in enumerate(ifcsts)]

    f_mean = [np.mean(f) for f in F_list]
    o_mean = [np.mean(o) for o in O_list]
    f_anom = [f-fmn for f,fmn in zip(F_list,f_mean)]
    o_anom = [o-omn for o,omn in zip(O_list,o_mean)]
    cov = np.array([np.sum(f*o) for f,o in zip(f_anom,o_anom)])
    f_var = np.array([np.sum(f**2) for f in f_anom])
    o_var = np.array([np.sum(o**2) for o in o_anom])
    f_std = np.sqrt(f_var)
    o_std = np.sqrt(o_var)
    std = f_std * o_std
    lac = cov / std
    return lac


def calc_apc(fcst,obs,varobj=None,latbounds=None,lonbounds=None):

    lats = varobj.lat
    lons = varobj.lon

    if latbounds is not None or lonbounds is not None:
        if latbounds is None:
            latbounds = (min(lats),max(lats))
        if lonbounds is None:
            lonbounds = (min(lons),max(lats))
        if min(lonbounds)<0:
            lons_shift = lons.copy()
            lons_shift[lons>180] = lons[lons>180]-360
            lons = lons_shift
        domain = np.where((lats>=min(latbounds)) & (lats<=max(latbounds)) & \
                          (lons>=min(lonbounds)) & (lons<=max(lonbounds)))
        fcst = np.array([f[domain] for f in np.array(fcst)])
        obs = np.array([o.squeeze()[domain] for o in np.array(obs)])
    else:
        fcst = np.array(fcst)
        obs = np.array(obs)
#        fcst = fcst.reshape([fcst.shape[0],np.product(fcst.shape[1:])])
#        obs = obs.reshape([obs.shape[0],np.product(obs.shape[1:])])

    num = np.sum(fcst * obs,axis=1)
    den = np.sqrt(np.sum(fcst**2,axis=1)*np.sum(obs**2,axis=1))
    apc = num / den

    return apc


#%%

def get_categorical_fcst(fcst,spread,bounds):
    fnorm = [stats.norm(loc = np.array(f),scale = np.array(s)) for f,s in zip(fcst,spread)]
    K = len(bounds)-1
    cat_fcst = [np.array([f.cdf(bounds[c+1])-f.cdf(bounds[c]) for c in range(K)]).squeeze() for f in fnorm]
    return cat_fcst

def get_categorical_obs(obs,bounds):
    K = len(bounds)-1
#    cat_obs = [np.sum([x>bounds[c] for c in range(1,K)],axis=0) for x in obs]
    cat_obs = [np.array([(x>bounds[c]) & (x<bounds[c+1]) for c in range(K)]) for x in obs]
    return cat_obs

def get_heidke(cat_fcst,cat_obs,weights=None,categorical=False):
    N,C = cat_fcst.shape
    if weights is None:
        weights = np.ones(N)
    N = sum(weights)
    if categorical:
        cat_fcst = np.round(cat_fcst)
    if N==0:
        H1 = 0
    else:
        H1 = N**-1 * np.sum(np.sum(cat_fcst*cat_obs,axis=1)*weights)
    H2 = 1/C
#    H2 = N**-2 * np.sum(np.sum((cat_fcst*cat_obs+(1-cat_fcst)*cat_obs)*weights[:,None],axis=1)*
#                       np.sum((cat_fcst*cat_obs+cat_fcst*(1-cat_obs))*weights[:,None],axis=1))
    HSS = (H1 - H2) / (1 - H2)
    return HSS

def get_rpss(cat_fcst,cat_obs,weights=None,categorical=False):
    N,C = cat_fcst.shape
    if weights is None:
        weights = np.ones(N)
    N = sum(weights)
    if categorical:
        cat_fcst = np.round(cat_fcst)
    BSref = (0.5-1)**2
    if N==0:
        BSfcst = 999
    else:
        BSfcst = (1/N)*sum((cat_fcst-cat_obs)[:,1]**2*weights)
    rpss = 1-(BSfcst/BSref)
    return rpss


#%%

def get_rhoinf_points(fcst,spread):
    if len(np.array(fcst).shape)<2:
        fcst = np.array(fcst).reshape(1,len(fcst))
        spread = np.array(spread).reshape(1,len(spread))
    f2 = [f**2 for f in fcst]
    e2 = [e**2 for e in spread]

    S2 = [f/e for f,e in zip(f2,e2)]
    rho_inf = np.array([s2 * ((s2+1)*s2)**-.5 for s2 in S2])
    if len(rho_inf)<2:
        rho_inf = rho_inf[0]
    return rho_inf

def get_rhoinf_time(fcst,spread,varobj=None,latbounds=None,lonbounds=None):

    lats = varobj.lat
    lons = varobj.lon
    if latbounds is None:
        latbounds = (np.amin(lats),np.amax(lats))
    if lonbounds is None:
        lonbounds = (np.amin(lons),np.amax(lons))
    if min(lonbounds)<0:
        lons_shift = lons.copy()
        lons_shift[lons>180] = lons[lons>180]-360
        lons = lons_shift
    domain = np.where((lats>=min(latbounds)) & (lats<=max(latbounds)) & \
                      (lons>=min(lonbounds)) & (lons<=max(lonbounds)))

    if len(np.array(fcst).shape)<2:
        fcst = np.array(fcst).reshape(1,len(fcst))
        spread = np.array(spread).reshape(1,len(spread))
    f2 = [f[domain]**2 for f in fcst]
    e2 = [e[domain]**2 for e in spread]

    S2 = [np.array(np.nansum(f)/np.nansum(e)) for f,e in zip(f2,e2)]
    rho_inf = np.array([s2 * ((s2+1)*s2)**-.5 for s2 in S2])
    if len(rho_inf)<2:
        rho_inf = rho_inf[0]
    return rho_inf

#%%

def map_proj(lats,lons):
    addcyc = False
    if np.amax(lons)-np.amin(lons)>345:
        addcyc = True
    if np.amin(lats)<0 and np.amax(lats)>0:
        m = Basemap(projection='cyl',llcrnrlat=np.amin(lats),urcrnrlat=np.amax(lats),\
                                     llcrnrlon=np.amin(lons),urcrnrlon=np.amax(lons),resolution='l')
    elif np.amin(lats)>=0 and np.amax(lats)>80 and np.amax(lons)-np.amin(lons)>345:
        m = Basemap(projection='npstere',lon_0=-90.,boundinglat=max([10,np.amin(lats)]),round=True,resolution='l')
    else:
        m = Basemap(projection='cyl',llcrnrlat=np.amin(lats),urcrnrlat=np.amax(lats),\
                                     llcrnrlon=np.amin(lons),urcrnrlon=np.amax(lons),resolution='l')
    return m,addcyc


def plot_map(z,varobj=None,levels=None,cmap=None,prop={}):

    import matplotlib.pyplot as plt

    default_prop={'figsize':(10,6),'cbarticklabels':None,'drawcountries':False,'drawstates':False}
    for key in prop.keys():
        default_prop[key] = prop[key]
    prop = default_prop

    zmap = varobj.regrid(z)

    if cmap is None:
        mycmap=make_colormap({0:'violet',
                          .22:'mediumblue',
                          .35:'lightskyblue',
                          .45:'w',
                          .55:'w',
                          .65:'gold',
                          .78:'firebrick',
                          1:'violet'})
    else:
        mycmap = cmap
    if levels is None:
        levels = np.linspace(-1, 1, 21)*np.amax(np.abs(z))
        cbticks = np.linspace(-1, 1, 11)*np.amax(np.abs(z))
    else:
        cbticks = levels[::2]

    m,addcyc = map_proj(varobj.lat,varobj.lon)

    fig = plt.figure(figsize = prop['figsize'],dpi=200)

    ax=plt.subplot()

    m.drawcoastlines(linewidth=0.5,color='0.5')
    if prop['drawcountries']:
        m.drawcountries(linewidth=0.5,color='0.5')
    if prop['drawstates']:
        m.drawstates(linewidth=0.5,color='0.5')

    lat = varobj.latgrid
    lon = varobj.longrid

    if addcyc:
        pltdata,lons = addcyclic(zmap,lon)
        lats = np.concatenate([lat,lat[:,0][:,None]],axis=1)
    else:
        pltdata,lons,lats = zmap,lon,lat
#    ipole = np.where(lats==90)
#    pltdata[ipole] = np.mean(pltdata[tuple([ipole[0]-1,ipole[1]])])
    mx,my = m(lons,lats)
    cbmap = m.contourf(mx, my, pltdata, cmap=mycmap,levels=levels,extend='both')
    plt.subplots_adjust(bottom=0.12)
    cax = fig.add_axes([0.15,0.05,0.7,0.03])

    cbar = fig.colorbar(cbmap,ticks=cbticks,cax=cax,orientation='horizontal')
    if prop['cbarticklabels'] is not None:
        cbar.ax.set_xticklabels(prop['cbarticklabels'])

    return ax

def timelonplot(ax,xlon,ytime,var,levels,cmap,FORECAST=None,datelab='on',lon0=0,cbar_label=None):
    import matplotlib.dates as mdates
    var=np.asarray(var)
    ilon0 = np.where(xlon<=lon0)[0][-1]
    mapvar=np.concatenate([var[:,ilon0:],var[:,:ilon0+1]],axis=1)
    xlon=np.concatenate([xlon,np.array([360])])
    cf = ax.contourf(xlon,ytime[0:len(mapvar)],mapvar,levels=levels,cmap=cmap,extend='both')
    xticks = np.arange(xlon[0],xlon[-1]+1,60)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int((i+lon0)%360) for i in xticks],fontsize=16)
    ax.yaxis_date()
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%-d-%b'))
    ax.axis([xlon[0],xlon[-1],ytime[-1],ytime[0]])
    ax.grid(color='0.5',linewidth=0.5)
    ax.set_xlabel('Longitude')
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    if datelab=='off':
        for item in ax.get_yticklabels():item.set_fontsize(0)
    if FORECAST is not None:
        fdates = [max(ytime)]+FORECAST['dates']
        ft0 = fdates[0]
        plt.plot(xticks,[ft0]*len(xticks),'k')
        plt.text(np.mean(xticks),ft0,'Begin Forecast',ha='center',va='top',fontsize=16)
        ax.axis([xlon[0],xlon[-1],fdates[-1],ytime[0]])
        initvar = mapvar[-1]
        mapvar = np.concatenate([FORECAST['var'][:,ilon0:],FORECAST['var'][:,:ilon0+1]],axis=1)
        mapvar = np.concatenate([initvar[None,:],mapvar],axis=0)
        ax.contourf(xlon,fdates,mapvar,levels=levels,cmap=cmap,extend='both')
    cbar=plt.colorbar(cf,orientation='horizontal',pad=0.1)
    if cbar_label is not None:
        cbar.ax.set_xlabel(cbar_label,fontsize=14)
    cbar.ax.tick_params(labelsize=14)

def timelatplot(ax,xlat,ytime,var,levels,cmap,FORECAST=None,datelab='on'):
    import matplotlib.dates as mdates
    var=np.asarray(var)
    ax.contourf(xlat,ytime[0:len(var)],var,np.linspace(vmin,vmax,256),cmap='RdBu_r')
    xticks = np.arange(min(xlat),max(xlat),15)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(i) for i in xticks],fontsize=16)
    ax.yaxis_date()
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%-d-%b'))
    ax.axis([min(xlat),max(xlat),ytime[-1],ytime[0]])
    ax.grid(color='k')
    ax.set_xlabel('Latitude')
    ax.set_title(title, fontsize=16)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    if datelab=='off':
        for item in ax.get_yticklabels():item.set_fontsize(0)
    if FORECAST is not None:
        fdates = [max(ytime)]+FORECAST['dates']
        ft0 = fdates[0]
        plt.plot(xticks,[ft0]*len(xticks),'k')
        plt.text(np.mean(xticks),ft0,'Begin Forecast',ha='center',va='top',fontsize=16)
        ax.axis([xlat[0],xlat[-1],fdates[-1],ytime[0]])
        initvar = mapvar[-1]
        mapvar = np.concatenate([FORECAST['var'][:,ilon0:],FORECAST['var'][:,:ilon0+1]],axis=1)
        mapvar = np.concatenate([initvar[None,:],mapvar],axis=0)
        ax.contourf(xlat,fdates,mapvar,levels=levels,cmap=cmap,extend='both')


def leave_one_out_bc(FCST,OBS):
    FCST_BC = []
    for i,F in enumerate(FCST):
        other_fcsts = np.concatenate(FCST[:i],FCST[i+1:],axis=0)
        other_obs = np.concatenate(OBS[:i],OBS[i+1:],axis=0)
        bias = np.mean(other_fcsts-other_obs,axis=0)
        FCST_BC.append(F-bias)
    return FCST_BC

def add_prop(input_prop,default_prop):

    r"""
    Overrides default property dictionary elements with those passed as input arguments.

    Parameters:
    -----------
    input_prop : dict
        Dictionary to use for overriding default entries.
    default_prop : dict
        Dictionary containing default entries.

    Returns:
    --------
    dict
        Default dictionary overriden by entries in input_prop.
    """

    #add kwargs to prop and map_prop
    for key in input_prop.keys(): default_prop[key] = input_prop[key]

    #Return prop
    return default_prop

def lat2str(x):
    deg = u"\u00B0"
    if x<0:
        return f'{abs(x)}{deg}S'
    else:
        return f'{x}{deg}N'

def lat2strNodeg(x):
    deg = u"\u00B0"
    if x<0:
        return f'{abs(x)}S'
    else:
        return f'{x}N'

def make_colormap(colors,whiten=0):

    z  = np.array(sorted(colors.keys()))
    n  = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)

    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        Ci = colors[z[i]]
        if type(Ci) == str:
            RGB = CC.to_rgb(Ci)
        else:
            RGB = Ci
        R.append(RGB[0] + (1-RGB[0])*whiten)
        G.append(RGB[1] + (1-RGB[1])*whiten)
        B.append(RGB[2] + (1-RGB[2])*whiten)

    cmap_dict = {}
    cmap_dict['red']   = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue']  = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)

    return mymap

def get_cmap_levels(colormap,levels):

    r"""
    Retrieve colormap instance and levels from input.

    Parameters
    ----------
    colormap : str or list
    levels : list
        List of contour levels. If length == 2, rcolormap is smooth and optimal ticks are returned

    Returns
    -------
    colors : cmap
        Matplotlib colormap object.
    levels : list
        List of contour levels.
    """

    #Matplotlib colormap name
    if isinstance(colormap,str):
        cmap = mlib.cm.get_cmap(colormap)

    #User defined list of colors
    elif isinstance(colormap,list):
        cmap = mcolors.ListedColormap(colormap)

    #Dictionary
    elif isinstance(colormap,dict):
        cmap = make_colormap(colormap)

    #Otherwise, a cmap was passed
    else:
        cmap = colormap

    #Normalize colors relative to levels
    norm = mcolors.Normalize(vmin=0, vmax=len(levels)-1)

    #If more than 2 levels were passed, use those for the contour levels
    if len(levels) > 2:
        colors = cmap(norm(np.arange(len(levels)-1)+.5))
        cmap = mcolors.ListedColormap(colors)

    #Otherwise, create a list of colors based on levels
    else:
        colors = cmap(norm(np.linspace(0,1,256)))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',colors)

        y0 = min(levels)
        y1 = max(levels)
        dy = (y1-y0)/8
        scalemag = int(np.log(dy)/np.log(10))
        dy_scaled = dy*10**-scalemag
        dc = min([1,2,5,10], key=lambda x:abs(x-dy_scaled))
        dc = .5*dc*10**scalemag
        c0 = np.ceil(y0/dc)*dc
        c1 = np.floor(y1/dc)*dc
        levels = np.arange(c0,c1+dc,dc)

        if scalemag > 0:
            levels = levels.astype(int)

    #Return colormap and levels
    return cmap, levels

# Interpolation function from x lat/lon grid to lIM variable object
def interp2LIM(lat,lon,z,varobj):
    from scipy.interpolate import interp2d
    if isinstance(z,np.ma.MaskedArray):
        z=z.filled(np.nan)
    f = interp2d(lon,lat,z)
    out = np.array([f(x,y)[0] for x,y in zip(varobj.lon,varobj.lat)])
    inan = np.where((varobj.lon<min(lon)) | (varobj.lon>max(lon)) | (varobj.lat<min(lat)) | (varobj.lat>max(lat)))
    out[inan]=0
    return out

def get_varpci(eof_lim,varname):
    pci = np.cumsum([eof_lim[key] for key in eof_lim.keys()])
    vari = list(eof_lim.keys()).index(varname)
    varpci = [pci[vari]-eof_lim[varname],pci[vari]]
    return varpci

def date_range_test(t,date_range,year_range=None):
    t_min,t_max = [dt.strptime(i,'%m/%d') for i in date_range]
    t_max += timedelta(days=1,seconds=-1)
    if t_min<t_max:
        test1 = (t>=t_min.replace(year=t.year))
        test2 = (t<=t_max.replace(year=t.year))
        tmp =  test1 & test2
    else:
        test1 = (t_min.replace(year=t.year)<=t<dt(t.year+1,1,1))
        test2 = (dt(t.year,1,1)<=t<=t_max.replace(year=t.year))
        tmp = test1 | test2
    if year_range is None:
        return tmp
    else:
        return tmp & (min(year_range)<=t.year<=max(year_range))

def lat2strNodeg(x):
    deg = u"\u00B0"
    if x<0:
        return f'{abs(x)}S'
    else:
        return f'{x}N'


#%%

def save_ncds(vardict,coords,attrs={},filename=None):

    r"""
    set up dictionary with vars and coords, including attributes

    {
    'coords': {'x': {'dims': ('x',), 'attrs': {}, 'data': []},
                'y': {'dims': ('y',), 'attrs': {},'data': []},
                'z': {'dims': ('x',), 'attrs': {}, 'data': []},},
    'attrs': {},
    'dims': {'x': 4, 'y': 5},
    'data_vars': {'foo': {'dims': ('x', 'y'),
                          'attrs': {},
                          'data': [] },}
    }

    """

    vardict_in = copy.deepcopy(vardict)
    coords_in = copy.deepcopy(coords)

    if 'time' in coords_in.keys():
        newtime = [np.double((t-dt(1800,1,1)).total_seconds()/3600) for t in coords_in['time']['data']]
        coords_in['time']['data'] = newtime

        if len(coords_in['time']['data'])>1:
            delta_t = np.gradient(newtime)[0]
            coords_in['time']['attrs'] = {'long_name':"Time",
                                   'delta_t':f"0000-00-{int(delta_t/24):02} {int(delta_t%24):02}:00:00",
                                   'standard_name':"time",
                                   'axis': "T",
                                   'units':"hours since 1800-01-01 00:00:0.0"}
        else:
            coords_in['time']['attrs'] = {'long_name':"Time",
                           'standard_name':"time",
                           'axis': "T",
                           'units':"hours since 1800-01-01 00:00:0.0"}

    if 'climo' in vardict_in.keys():
        try:
            long_name = vardict_in['climo']['attrs']['long_name']
            vardict_in['climo']['attrs']['long_name'] = 'Climatology of '+long_name
        except:
            vardict_in['climo']['attrs']['long_name'] = 'Climatology'

    encoding = {k: {'dtype': 'double', '_FillValue': 1e30} for k in coords_in.keys()}
    for k in vardict_in.keys():
        encoding.update({k: {'dtype': 'single', '_FillValue': 1e30}})

    ds = xr.Dataset.from_dict({
        'coords':coords_in,
        'data_vars':vardict_in,
        'dims':[k for k,v in coords_in.items()],
        'attrs':attrs,
    })
    for variable in ds.variables.values():
        # Some units are None Type that would cause errors when writing out as netcdf
        variable.attrs = {key: value for key, value in variable.attrs.items() if value is not None}
        # for k, v in variable.attrs.items():
        #     print(k, v)
        #     print(type(k),type(v))
    if isinstance(filename,str):
        ds.to_netcdf(filename,encoding=encoding)
        ds.close()
    else:
        print('filename must be a string')
