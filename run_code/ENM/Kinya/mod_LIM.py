import warnings
import numbers
import numpy as np
import pandas as pd
import xarray as xr
from scipy import linalg
xr.set_options(keep_attrs=True)

def reset_data_shape(x_tr, mask, x_shape):
    """
    Reset data shape from (time, space) to (time, y, x, ..)
    
    Parameters
    ----------
    x_tr : (n, ns) array
            truncated array. ns <= nx*ny 
            
    mask   : (nx*ny) bool array
            True if no value
              
    x_shape : tuple
            Original shape of x

    Returns
    -------
    x: (n, ny, nx, ..)
    """
    n = x_tr.shape[0]
    
    x = np.empty((n,) + x_shape).reshape(n, -1)
    x[:] = np.nan
    
    x[:, ~mask] = x_tr
    
    return x.reshape((n,) + x_shape)

def set_num_eigs_vars(num_eigs, all_var_lst):
    """
    Set the lists of number of eigenvalues and variables
    
    Parameters
    ----------    
    num_eigs: int or pandas Series or pandas DataFrame
        Number of eigenvalues/vectors.
        Example for mixed single and multivariate EOFs:
            pd.DataFrame({'ne': [20, 40], 
                          'variables': [['sf850', 'sf250'], 'olr']},
                          index = ['sf', 'olr'])
    all_var_lst: list
        list of all variables
        
    Returns
    -------
    ne_lst: ndarray
        list of number of eigenvalues   
        
    var_lst: list
        list of variables used for EOFs 
        Nested list for multi-variate EOFs
    
    new_var_lst: list
        list of new variables after EOFs
    """
    
    if isinstance(num_eigs, numbers.Integral):
        var_lst = all_var_lst
        new_var_lst = all_var_lst
        ne_lst = np.array([num_eigs] * len(var_lst))
    elif isinstance(num_eigs, pd.Series):
        var_lst = all_var_lst
        new_var_lst = all_var_lst
        ne_lst = num_eigs.values
    elif isinstance(num_eigs, pd.DataFrame):    # for multivariate EOFs
        var_lst = num_eigs['variables'].values  # before EOFs
        new_var_lst = num_eigs.index            # after EOFs
        ne_lst = num_eigs['ne'].values
    else:
        raise ValueError('num_eigs should be integer or pandas Series/DataFrame')
        
    return ne_lst, var_lst, new_var_lst

def calc_pc_eof(ds, num_eigs):
    """
    Calculate standarized PCs and corresponding EOFs for leading num_eigs modes.
    Each field is standarlized by the domain-integrated variance
    num_eigs modes. 
    
    Parameters
    ----------
    ds : xarray dataset
        Dataset to calculate EOFs from. Each variable has a shape of (nt, ny, nx)
        
    num_eigs: int or pandas Series or pandas DataFrame
        Number of eigenvalues/vectors to return.  Must be less than min(nt, ny*nx).
        Example for mixed single and multivariate EOFs:
            pd.DataFrame({'ne': [20, 40], 
                          'variables': [['sf850', 'sf250'], 'olr']},
                          index = ['sf', 'olr'])

    Returns
    -------
    pc : xarray dataset
        standarized PCs (num_eigs, nt)
        
    eof : xarray dataset
        Leading EOFs
    
    std_factor : xarray dataset
        Standarization factor.
        Domain-integrated variance
    
    pca_score : xarray dataset
    """
    pca_score_len = min(ds.time.size, 100)
    
    # Weight the data with sqrt(cos(phi))
    wgt = np.sqrt(np.cos(np.deg2rad(ds.lat)))
    ds_wgt = ds * wgt

    ne_lst, var_lst, new_var_lst = set_num_eigs_vars(num_eigs, list(ds.keys()))
            
    ne_max = max(ne_lst)
    pc_dict = {}
    eof_lst = []
    pcstd_dict = {}
    pca_score_dict = {}
    varstd_lst = []

    # Standarize
    std_factor = ds_wgt.var(dim='time').sum(dim=['lat', 'lon']) ** 0.5
    ds_std_wgt = ds_wgt / std_factor

    for ne, var, new_var in zip(ne_lst, var_lst, new_var_lst):
        if isinstance(var, list): # multi-variate
            da = ds_std_wgt[var].to_array().transpose('time', ...)
        else:
            da = ds_std_wgt[var]
        
        # flatten to (time, space)
        x = da.data.reshape(da.shape[0], -1)
        
        # find grids containing nans
        mask = np.any(np.isnan(x), axis=0)
        x_ma = x[:, ~mask]
        #print(x_ma.shape)
        
        # eof
        eofs, sigma, VT = np.linalg.svd(x_ma.T, full_matrices=False)
        smat = np.diag(sigma)
        pcs = smat @ VT
        pca_score = sigma**2 / (sigma**2).sum()
        
        # 1D space -> 2D lat, lon
        eofs_2d = reset_data_shape(eofs.T, mask, da.shape[1:])
        
        # Truncate
        pcs[ne:] = np.nan
        eofs_2d[ne:] = np.nan
        
        # Make xarray
        dims = ('mode',) + da.dims[1:]
        coords = {key: da.coords[key] for key in da.coords if key not in ['time']}
        coords['mode'] = np.arange(ne_max)
        eof_da = xr.DataArray(eofs_2d[:ne_max], dims=dims, coords=coords,
                              name=new_var)

        if isinstance(var, list):
            eof_da = eof_da.to_dataset('variable')
            for each_var in var:
                eof_da[each_var].attrs = ds[each_var].attrs
        else:
            eof_da.attrs = ds[var].attrs

        # Append
        pc_dict.update({new_var: (('mode', 'time'), pcs[:ne_max])})
        eof_lst.append(eof_da)
        #pcstd_dict.update({new_var: (pcstd)})
        pca_score_dict.update({new_var: (('mode'), pca_score[:pca_score_len])}) # first 300s
        
    # Make xarray         
    pc = xr.Dataset(pc_dict,
            coords = {'time': ds.time, 'mode': range(ne_max)})
    
    eof_wgt = xr.merge(eof_lst)
    
    #std_factor = xr.Dataset(pcstd_dict)
    
    pca_score = xr.Dataset(pca_score_dict,
                           coords={'mode': range(pca_score_len)})
    
    # Reverse the weights
    eof = eof_wgt / wgt
    
    return pc, eof, std_factor, pca_score

def proj_eof(ds, eof, num_eigs, std_factor):
    """
    Project x onto EOF space and obtain standarized PCs
    Z = E^T X
    
    Parameters
    ----------
    ds : xarray dataset
        Dataset to project. Each variable has a shape of (..., ny, nx)
        
    eof : xarray dataset
        Each variable has a shape of (ne_max, ny, nx)
        
    num_eigs: int or pandas Series or pandas DataFrame
        Number of eigenvalues/vector
        
    std_factor : xarray dataset
        Standarization factor

    Returns
    -------
    z : xarray dataset
        standarized PCs (ne, :, ...)
    """
    ds = ds.sel(lat=eof.lat, lon=eof.lon)
    
    # Weight the data with sqrt(cos(phi))
    wgt = np.sqrt(np.cos(np.deg2rad(ds.lat)))
    ds_std_wgt = ds * wgt / std_factor
    eof_wgt = eof * wgt   

    ne_lst, var_lst, new_var_lst = set_num_eigs_vars(num_eigs, list(eof.keys()))
    ne_max = max(ne_lst)
    
    dict1 = {}
    for ne, var, new_var in zip(ne_lst, var_lst, new_var_lst):
        if isinstance(var, list): # multi-variate
            # normalize each field
            da = ds_std_wgt[var].to_array().transpose(..., 'variable', 'lat', 'lon')
            eof_da = eof_wgt[var].to_array().transpose('mode', ...)
            dim = ('mode',) + da.dims[:-3]
            z_shape = (ne_max,) + da.shape[:-3]
        else:
            da = ds_std_wgt[var]
            eof_da = eof_wgt[var]
            dim = ('mode',) + da.dims[:-2]
            z_shape = (ne_max,) + da.shape[:-2]

        # reshape
        eT = np.nan_to_num(eof_da.data.reshape(ne_max, -1))
        x = np.nan_to_num(da.data.reshape(-1, eT.shape[1])).T

        z1 = eT @ x

        # Truncate
        z1[ne:] = np.nan

        # 
        dict1.update({new_var: (dim, z1.reshape(z_shape))})

    # Make xarray
    coords = {key: ds.coords[key] for key in ds.coords 
              if key not in ['lat', 'lon']}
    coords['mode'] = np.arange(ne_max)
    
    z = xr.Dataset(dict1, coords=coords)
    
    return z    
    
def proj_ori(pc, eof, num_eigs, std_factor):
    """
    Project back to the original space
    X = E Z * sigma
    
    Parameters
    ----------
    pc : xarray dataset
        standarized PCs (ne, ...)

    eof : xarray dataset
        Each variable has a shape of (ne, ny, nx)
        
    num_eigs: int or pandas Series or pandas DataFrame
        Number of eigenvalues/vector
        
    std_factor : xarray dataset
        Standarization factor
        
    Returns
    -------
    x : xarray dataset
        (Truncated) data in the original space.
    """
    pc = pc.transpose('mode', ...)
    coords1 = {key: pc.coords[key] for key in pc.coords if key != 'mode'}

    ne_lst, var_lst, new_var_lst = set_num_eigs_vars(num_eigs, list(eof.keys()))
    ne_max = max(ne_lst)
    
    lst = []
    for ne, var, new_var in zip(ne_lst, var_lst, new_var_lst):
        if isinstance(var, list): # multi-variate
            eof_da = eof[var].to_array().transpose('mode', ...)
        else:
            eof_da = eof[var]
                    
        # reshape
        e = eof_da.data.reshape(ne_max, -1).T
        z = pc[new_var].data.reshape(ne_max, -1)
        
        # Reconstruct fields
        tmp = e[:, :ne] @ z[:ne, :]

        # Make xarray
        dims = pc[new_var].dims[1:] + eof_da.dims[1:]
        x_shape = pc[new_var].shape[1:] + eof_da.shape[1:]     
        coords2 = {key: eof_da.coords[key] for key in eof_da.coords if key != 'mode'}
        
        x_da = xr.DataArray(tmp.T.reshape(x_shape), dims=dims, 
                            coords={**coords1, **coords2},
                            name=new_var)  
    
        if isinstance(var, list):
            x_da = x_da.to_dataset('variable')
            for each_var in var:
                x_da[each_var].attrs = eof[each_var].attrs
        else:
            x_da.attrs = eof[var].attrs

        # Append
        lst.append(x_da)
    
    x = xr.merge(lst) * std_factor
    
    return x

def proj_x_to_z(x, evects):
    """
    Project x to the given eigenvector space
    
    Parameters
    ----------
    x : xarray
        State vector (n_features,)
        
    evects : xarray
        eigenvectors (n_features, n_modes)

    Returns
    -------
    z : xarray
        1D or 2D state vector (n_modes,)
    """
    
    n_features = x.shape[0]
    
    z = np.linalg.inv(evects) @ x.data.reshape(n_features, -1)
    
    dims = evects.dims[1:]
    coords = {key: evects.coords[key] for key in dims}
    
    if len(x.shape) > 1:
        dims += x.dims[1:]
        coords.update({key: x.coords[key] for key in x.dims[1:]})
        
    z = xr.DataArray(z.reshape(x.shape), dims=dims, coords=coords)
    
    return z

def proj_z_to_x(z, evects):
    """
    Project z back to x given eigenvector space
    X = EZ
    
    Parameters
    ----------
    z : xarray
        1D or 2D state vector (n_modes,)
        
    evects : xarray
        eigenvectors (n_features, n_modes)

    Returns
    -------
    x : xarray
        1D or 2D state vector (n_features,)
    """
    
    n_features = z.shape[0]
    
    x = (evects.data @ z.data.reshape(n_features, -1)).real    # imagnary part should be 0

    dims = evects.dims[:1]
    coords = {key: evects.coords[key] for key in dims}
    
    if len(z.shape) > 1:
        dims += z.dims[1:]
        coords.update({key: z.coords[key] for key in z.dims[1:]})

    x = xr.DataArray(x.reshape(z.shape), dims=dims, coords=coords)
    
    return x
#def x_subspace(x, evects):
#    """
#    Estimate the subspace of x given eigenvectors
#    
#    Parameters
#    ----------
#    x : xarray
#        1D or 2D state vector (n_features,)
#        
#    evects : xarray
#        eigenvectors (n_features, n_features)
#        
#    Returns
#    -------
#    x_sub : xarray
#        1D or 2D state vector (n_features,)
#    """
#    
#    n_features = x.shape[0]
#    
#    z = np.linalg.inv(evects) @ x.data.reshape(n_features, -1)
#    
#    x_sub = (evects @ z).real    # imagnary part should be 0
#    
#    dims = evects.dims[1:]
#    coords = {key: evects.coords[key] for key in dims}
#    
#    if len(x.shape) > 1:
#        dims += x.dims[1:]
#        coords.update({key: x.coords[key] for key in x.dims[1:]})
#        
#    x_sub = xr.DataArray(x_sub.reshape(x.shape), dims=dims, coords=coords)
#    
#    return x_sub
# ------------------------------------------------------------------------------
#  LIM
# ------------------------------------------------------------------------------

def pc_to_state(pc):
    """
    Stack PCs of different variables and return a state vector
    
    Parameters
    ----------
    pc : xarray dataset
         Principle components (ne, ...)
    
    Returns
    -------
    x : xarray dataset
        State vector used in LIM
    """
    
    x = pc.to_array().stack(feature=('variable', 'mode')).T
    x = x.dropna(dim='feature')   # Drop nans for LIMs with unequal numbers of EOF
    
    return x

def state_to_pc(x):
    """
    Unstack state vector and return PCs
    
    Parameters
    ----------
    x : (nk, ...) xarray dataset
        State vector used in LIM
        nk is the number of feature
        
    Returns
    -------
    pc : xarray dataset
        (num_eigs, ...)
    """
    
    pc = x.to_unstacked_dataset('feature')
    
    return pc

def calc_lim_g(x, tau, tunit='D'):
    """
    Calculate the LIM propagator G
    
    x(t+tau) = G * x(tau) + noise
    
    Parameters
    ----------
    x : xarray
        2D state vector (n_features, n_samples)
        The sample dimension should be time.
        
    tau : int 
        lag time calculating G for.
    
    tunit : str, default `D`
        unit of tau. 

    Returns
    -------
    G : (n_features, n_features) ndarray
        LIM propagator
    """
    
    t0 = pd.to_datetime(x.time.data)
    #t1 = t0 + pd.Timedelta(tau, unit=tunit)
    if tunit == 'D':
        t1 = t0 + pd.DateOffset(days=tau)
    elif tunit == 'M':
        t1 = t0 + pd.DateOffset(months=tau)
    elif tunit == 'Y':
        t1 = t0 + pd.DateOffset(years=tau)
    else:
        raise ValueError('tunit not supported')

    # Find indices where t1 exist
    #bool_t0 = [d in t0 for d in t1]  
    #bool_t1 = [d in t1 for d in t0]
    bool_t0 = np.isin(t1, t0)
    bool_t1 = np.isin(t0, t1)
    # print(f'Data used to estimate G: {np.sum(bool_t0)} / {len(t0)}')
    
    # Calculate covariance 
    x0 = x.sel(time=bool_t0).data
    xt = x.sel(time=bool_t1).data
    x0x0 = x0 @ x0.T
    xtx0 = xt @ x0.T

    #G = xtx0 @ np.linalg.inv(x0x0)
    G = xtx0 @ np.linalg.pinv(x0x0)
    
    # Calculate eigenvalues to check that all modes are damped
    g_evals, evects = np.linalg.eig(G)
    l_evals = (1./tau) * np.log(g_evals)

    if np.any(l_evals.real >= 0):
        #raise ValueError('Positive eigenvalues detected in forecast matrix L.')
        warnings.warn('Positive eigenvalues detected in forecast matrix L.')

    return G

#def calc_lim_q(G, x, tau):


def calc_lim_fcst(x, tau0, G, n_leads):
    """
    Deterministic forecast of x
    
    Parameters
    ----------
    x : xarray
        1D or 2D state vector (n_features,)
        Initial conditions

    tau0 : int 
        lag time used to calculate G
        
    G : (n_features, n_features) ndarray
        LIM propagator

    n_leads : int
       lead time

    Returns
    -------
    xf : (n_leads + 1, n_features,) xarray
        Forecasts of x, including initial conditions
    """    
    tmp = x.expand_dims({'lead': np.arange(n_leads+1)})
    xf = np.zeros(tmp.shape)
    xf[0] = x.data
    
    L = linalg.logm(G) / tau0

    for j in range(1, n_leads+1):
        xf[j] = linalg.expm(L * j) @ x.data
    
    xf = xr.DataArray(xf, dims=tmp.dims, coords=tmp.coords)
    
    return xf

def calc_lim_fcstb(x, tau0, G, n_leads):
    """
    Deterministic forecast of x
    backward in time
    
    Parameters
    ----------
    x : xarray
        1D or 2D state vector (n_features,)
        Initial conditions

    tau0 : int 
        lag time used to calculate G
        
    G : (n_features, n_features) ndarray
        LIM propagator

    n_leads : int
       lead time

    Returns
    -------
    xf : (n_leads + 1, n_features,) xarray
        Forecasts of x, including initial conditions
    """    
    tmp = x.expand_dims({'lead': -np.arange(n_leads+1)})
    xf = np.zeros(tmp.shape)
    xf[0] = x.data
    
    L = linalg.logm(G) / tau0

    for j in range(1, n_leads+1):
        xf[j] = linalg.expm(-L * j) @ x.data
    
    xf = xr.DataArray(xf, dims=tmp.dims, coords=tmp.coords)
    
    return xf

def calc_lim_fcst_eig(x, tau0, G, n_leads):
    """
    Forecast x after projecting onto eigen space
    
    Parameters
    ----------
    x : xarray
        1D or 2D state vector (n_features,)
        Initial conditions

    tau0 : int 
        lag time used to calculate G
        
    G : (n_features, n_features) ndarray
        LIM propagator

    n_leads : int
       lead time

    Returns
    -------
    xf : (n_leads + 1, n_features,) xarray
        Forecasts of x, including initial conditions
    """    
    tmp = x.expand_dims({'lead': np.arange(n_leads+1)})
    xf = np.zeros(tmp.shape)
    xf[0] = x.data
    
    g_evals, evects = np.linalg.eig(G)
    l_evals = np.log(g_evals) / tau0
    
    # project x to eigen space
    z = np.linalg.inv(evects) @ x.data
        
    for j in range(1, n_leads+1):
        zf = np.diag(np.exp(j * l_evals)) @ z  # forecast
        xf[j] = (evects @ zf).real          # imagnary part should be 0

    xf = xr.DataArray(xf, dims=tmp.dims, coords=tmp.coords)

    return xf

def calc_lim_fcst_eig_sgls_mod(x, tau0, G, n_leads):
    """
    Forecast x after projecting onto eigen space
    using a single ENM by modifying the model
    
    Parameters
    ----------
    x : xarray
        1D or 2D state vector (n_features,)
        Initial conditions

    tau0 : int 
        lag time used to calculate G
        
    G : (n_features, n_features) ndarray
        LIM propagator

    n_leads : int
       lead time

    Returns
    -------
    xf : (n_features, n_leads+1, n_features,) xarray
        Forecasts of x, including initial conditions
        
    """    
    
    tmp = x.expand_dims({'enm':  np.arange(x.shape[0]), 
                         'lead': np.arange(n_leads+1)})
    xf = np.zeros(tmp.shape)
    xf[:, 0] = x.data[np.newaxis]
    
    g_evals, evects = np.linalg.eig(G)
    l_evals = np.log(g_evals) / tau0
    
    # project x to eigen space
    z = np.linalg.inv(evects) @ x.data
    
    for k in range(x.shape[0]):
        # Copy eigenvalues
        l_evals_mod = np.copy(l_evals)
        
        # Turn off other modes
        l_evals_mod[:k] = 0
        l_evals_mod[k+1:] = 0
        
        for j in range(1, n_leads+1):
            zf = np.diag(np.exp(j * l_evals_mod)) @ z # forecast
            xf[k, j] = (evects @ zf).real    # imagnary part should be 0

    xf = xr.DataArray(xf, dims=tmp.dims, coords=tmp.coords)

    return xf

def calc_lim_fcst_eig_sgls_ic(x, tau0, G, n_leads):
    """
    Forecast x after projecting onto eigen space
    using a single ENM by modifying ICs
    
    Parameters
    ----------
    x : xarray
        1D or 2D state vector (n_features,)
        Initial conditions

    tau0 : int 
        lag time used to calculate G
        
    G : (n_features, n_features) ndarray
        LIM propagator

    n_leads : int
       lead time

    Returns
    -------
    xf : (n_features, n_leads+1, n_features,) xarray
        Forecasts of x, including initial conditions
        
    """    
    tmp = x.expand_dims({'enm':  np.arange(x.shape[0]), 
                         'lead': np.arange(n_leads+1)})
    xf = np.zeros(tmp.shape)
    xf[:, 0] = x.data[np.newaxis]
    
    g_evals, evects = np.linalg.eig(G)
    l_evals = np.log(g_evals) / tau0
    
    # project x to eigen space
    z = np.linalg.inv(evects) @ x.data
    
    for k in range(x.shape[0]):
        # copy ICs
        z_mod = np.copy(z)
        
        # Turn off other modes
        z_mod[:k] = 0
        z_mod[k+1:] = 0        
        
        for j in range(1, n_leads+1):
            zf = np.diag(np.exp(j * l_evals)) @ z_mod # forecast
            xf[k, j] = (evects @ zf).real    # imagnary part should be 0
            
    xf = xr.DataArray(xf, dims=tmp.dims, coords=tmp.coords)

    return xf

def calc_enms(x, G, tau0, eof, num_eigs, std_factor):
    """
    Estimate ENMs
    
    Parameters
    ----------
    x : xarray
        2D state vector (n_features, n_samples)
        
    G : (n_features, n_features) ndarray
        LIM propagator

    tau0 : int 
        lag time used to calculate G
        
    eof : xarray dataset
        Each variable has a shape of (ne, ny, nx)
        
    num_eigs: int or pandas Series or pandas DataFrame
        Number of eigenvalues/vector
        
    std_factor : xarray dataset
        Standarization factor

    Returns
    -------
    enm : xarray dataset
    
    eig_vecs : xarray dataset
    """
    
    g_evals, evects = np.linalg.eig(G)
    omega = np.arctan(g_evals.imag / g_evals.real) / tau0
    e_folding = -tau0 / np.log(np.absolute(g_evals)) 

    evects = xr.DataArray(
        evects, dims=['feature', 'enm'], 
        coords={'feature': x.coords['feature'], 
                'enm': np.arange(evects.shape[1]) + 1}
    )
    
    # project eigenvectors to orignal space
    evects_pc = state_to_pc(evects)
    
    enm = proj_ori(evects_pc, eof, num_eigs, std_factor)
    
    # Combine other info
    dict1 = {'g_eval': (('enm'), g_evals), 
               'freq': (('enm'), omega), 
             'period': (('enm'), 2*np.pi/omega), 
          'e_folding': (('enm'), e_folding)}
    
    return enm.update(dict1), evects

def calc_optimal_growth(L, n, tau):
    """
    Given the LIM propagator L and the target state n, 
    calculate optimal intial condition and growth rate
    
    Parameters
    ----------
    L : (n_features, n_features) ndarray
        LIM propagator
        
    n : xarray dataset
        Target state vector (1D)
        
    tau : int 
        Target growth period
        
    Returns
    -------
    p : xarray dataset
        Optimal IC (1D)
        
    mu : float
        Maximum growth rate    
    """

    eps = 1e-5

    n = n / np.linalg.norm(n)
    #N = np.diag(n**2) 
    N = np.array([n]).T @ np.array([n]) + eps * np.eye(n.shape[0])
    
    if np.linalg.det(N) == 0:
       raise ValueError('Norm kernel is singular. Change epsilon.')
    
    G_tau = linalg.expm(L * tau)
    
    A = G_tau.T @ N @ G_tau
    mus, ps = np.linalg.eig(A)
    
    idx = mus.argmax()
    
    p = xr.DataArray(ps[:, idx], dims=n.dims, coords=n.coords)
    
    ## Adjust sign
    p_fcst = G_tau @ p.data

    if n.data @ p_fcst > 0:
        return p, mus[idx]
    else:
        return -p, mus[idx]
    
def calc_norm(x, n):
    """
    Calculate norm using a specified weight
    
    Parameters
    ----------
    x : xarray
        state vector (n_features,)
        
    n : xarray dataset
        Weights (1D)
        
    Returns
    -------
    norm : xarray
    """
    eps = 1e-5

    n = n / np.linalg.norm(n)
    N = np.array([n]).T @ np.array([n]) + eps * np.eye(n.shape[0])
    
    if np.linalg.det(N) == 0:
        raise ValueError('Norm kernel is singular. Change epsilon.')
    
    N = xr.DataArray(N, dims=['feature', 'feature1'])
    
    norm = xr.dot((x @ N).rename({'feature1': 'feature'}), x, dims='feature')
        
    return norm
    
def calc_sv(L, w_0, w_tau, tau):
    """
    Given the linear propagator L and weights w_0/w_tau, 
    calculate singular vectors and growth rate
    
    Parameters
    ----------
    L : (n_features, n_features) ndarray
        LIM propagator
        
    w_0 : xarray dataset
        Initial weights (1D)

    w_tau : xarray dataset
        Final weights (1D)
        
    tau : int 
        Target growth period
        
    Returns
    -------
    p : xarray dataset
        Optimal IC (1D)
        
    mu : float
        Maximum growth rate    
    """
    eps = 1e-5

    w_0 = w_0 / np.linalg.norm(w_0)
    w_tau = w_tau / np.linalg.norm(w_tau)

    W_0 = np.array([w_0]).T @ np.array([w_0]) + eps * np.eye(w_0.shape[0])
    W_tau = np.array([w_tau]).T @ np.array([w_tau]) + eps * np.eye(w_tau.shape[0])
    
    if np.linalg.det(W_0) == 0:
        raise ValueError('Norm kernel is singular. Change epsilon.')
        
    W_0_inv = np.linalg.pinv(W_0)
    
    G_tau = linalg.expm(L * tau)
    
    A = W_0_inv @ G_tau.T @ W_tau @ G_tau
    mus, ps = np.linalg.eig(A)
    
    idx = mus.argmax()
    
    p = xr.DataArray(ps[:, idx], dims=w_0.dims, coords=w_0.coords)
    
    ## Adjust sign
    p_fcst = G_tau @ p.data

    if w_tau.data @ p_fcst > 0:
        return p, mus[idx]
    else:
        return -p, mus[idx]
    
# ------------------------------------------------------------------------------
#  Calculate errors, correlation of forecasts
# ------------------------------------------------------------------------------

def calc_stats(fcst, ref, tunit='D'):
    """
    Calculate normalized area-weighted MSE, 
              area-weighted uncentered anomaly correlation 
    
    Parameters
    ----------
    fcst : xarray dataset (time, lead, lat, lon)
        Forecasted data.
        
    ref : xarray dataset (time, lat, lon)
        Reference data.

    tunit: str, default `D`
        unit of lead

    Returns
    -------
    mse, acu : xarray dataset
    
    """
    # Here the weight is cos(phi) 
    wgt = np.cos(np.deg2rad(ref.lat))

    t0 = fcst.time.data
    t_ref = ref.time.data
    
    lst1 = []
    lst2 = []
    for lead in fcst.lead.data:
        t1 = t0 + pd.Timedelta(lead, unit=tunit)
        t_in = np.intersect1d(t_ref, t1)  ## intersecting times
        
        x = fcst.sel(lead=lead).assign_coords(time=t1).sel(time=t_in)
        y = ref.sel(time=t_in)
        #print('Lead = {}, sample size = {}'.format(lead, t_in.shape[0]))
        
        ## Calculate metrics
        mse = ((x - y) ** 2).weighted(wgt).mean(dim=['lat', 'lon'])
        
        acu = (x * y).weighted(wgt).sum(dim=['lat', 'lon']) \
            / ((x**2).weighted(wgt).sum(dim=['lat', 'lon']) \
             * (y**2).weighted(wgt).sum(dim=['lat', 'lon'])) ** 0.5
        
        lst1.append(mse)
        lst2.append(acu)
        
    # Climatological variance
    ref_var = ((ref**2).weighted(wgt).mean(dim=['lat', 'lon'])).mean(dim='time')
    mse = xr.concat(lst1, dim='lead') / ref_var
    acu = xr.concat(lst2, dim='lead')
    
    return mse, acu

def calc_pnt_stats(fcst, ref, tunit='D'):
    """
    Calculate skill metrics at each grid point
    
    Parameters
    ----------
    fcst : xarray dataset (time, lead, lat, lon)
        Forecasted data.
        
    ref : xarray dataset (time, lat, lon)
        Reference data.

    tunit: str, default `D`
        unit of lead

    Returns
    -------
    mse, ce, acc, acu : xarray dataset
    
    """

    t0 = fcst.time.data
    t_ref = ref.time.data
    
    lst = []
    for lead in fcst.lead.data:
        t1 = t0 + pd.Timedelta(lead, unit=tunit)
        t_in = np.intersect1d(t_ref, t1)  ## intersecting times
        
        x = fcst.sel(lead=lead).assign_coords(time=t1).sel(time=t_in)
        y = ref.sel(time=t_in)
        x_m = x.mean(dim='time')
        y_m = y.mean(dim='time')
        
        ## Calculate metrics
        mse = ((x - y) ** 2).mean(dim='time')
        
        ce = 1 - mse / ((y - y_m) ** 2).mean(dim='time')
        
        acc = ((x - x_m) * (y - y_m)).sum(dim='time') \
            / (((x - x_m)**2).sum(dim='time') \
             * ((y - y_m)**2).sum(dim='time')) ** 0.5
        
        acu = (x * y).sum(dim='time') \
            / ((x**2).sum(dim='time') \
             * (y**2).sum(dim='time')) ** 0.5
        
        lst.append((mse, ce, acc, acu))
        
    mse = xr.concat([row[0] for row in lst], dim='lead')
    ce  = xr.concat([row[1] for row in lst], dim='lead')
    acc = xr.concat([row[2] for row in lst], dim='lead')
    acu = xr.concat([row[3] for row in lst], dim='lead')
    
    return mse, ce, acc, acu

# ------------------------------------------------------------------------------
#  AR1
# ------------------------------------------------------------------------------

def calc_ar1_cal(ds, n_leads, tunit='D'):
    """
    Calibrate data for AR1 process
    
    x(t+tau) = rho(tau) * x(tau) + noise
    
    Parameters
    ----------
    ds : (nt,) xarray
        first dimension should be time
    
    n_leads : int 
        lag time
    
    tunit : str, default `D`
        unit of lead. 

    Returns
    -------
    rho : (n_leads,) xarray
    
    """
    t0 = ds.time.data
    nt = ds.time.size
    
    dict1 = {}
    for var in list(ds.keys()):
        x_nd = ds[var].data

        # Reshape to (time, space)
        x = x_nd.reshape(x_nd.shape[0], -1)

        corr = np.empty((n_leads + 1, x.shape[1]))
        corr[:] = np.nan

        for i in range(n_leads + 1):
            t1 = t0 + pd.Timedelta(i, unit=tunit)

            # Find indices where t1 exist
            #bool_t0 = [d in t0 for d in t1]  
            #bool_t1 = [d in t1 for d in t0]
            bool_t0 = np.isin(t1, t0)
            bool_t1 = np.isin(t0, t1)
    
            idx0 = np.arange(nt)[bool_t0]
            idx1 = np.arange(nt)[bool_t1]
            
            for j in range(x.shape[1]):
                corr[i, j] = np.corrcoef(x[idx0, j], x[idx1, j])[0, 1]
        
        # Reshape back to original dimensions
        dim = ('lead',) + ds[var].dims[1:]
        dict1.update({var: (dim, corr.reshape((n_leads+1,) + x_nd.shape[1:]))})
        
    # Make xarray
    coords = {key: ds.coords[key] for key in ds.coords if key != 'time'}
    coords['lead'] = np.arange(n_leads+1)
    
    rho = xr.Dataset(dict1, coords=coords)
    
    return rho

def calc_ar1_fcst(ds, rho):
    """
    Forecast AR1 process
    
    <x(t+tau)> = rho(tau) * x(tau)
    
    Parameters
    ----------
    ds : xarray
        Initial conditions
    
    rho : xarray

    Returns
    -------
    fcst : xarray
    
    """
    fcst = ds * rho
    
    return fcst
