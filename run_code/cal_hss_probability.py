import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from lib import driver
# from lib import plot 
from lib.tools import get_categorical_obs, get_categorical_fcst, get_heidke, get_rpss
# from lib import verif
from lib.tools import *
import warnings

def check_lat_order(dataset,verbose=False):
    """
    Check the order of latitude coordinates in the dataset.
    
    Parameters:
    - dataset: xarray.Dataset or xarray.DataArray
    
    Returns:
    - True if latitude is ordered from North to South (N to S).
    - False if latitude is ordered from South to North (S to N).
    - None if the order is ambiguous or the dataset is empty.
    """
    if 'latitude' in dataset.coords:
        dataset = dataset.rename_dims({'latitude': 'lat', 'longitude': 'lon'})

    lat_coords = dataset['lat']
    lat_diff = lat_coords.diff(dim='lat')
    if verbose:
        print(lat_diff)

    if (lat_diff < 0.).all():
        dataset= dataset.sel(lat=dataset.lat[::-1])
        print('change latitude from S to N')
        # print(dataset.lat)
    elif (lat_diff > 0.).all():
        dataset = dataset
        print('====latitude from S to N----')
        # print(dataset.lat)
    else:
        print('!!!!!! Latitude ambiguous or unordered !!!!!!')
    return(dataset)





varname = 'T2m'
anomvar = varname+'_anom'
spreadvar = varname+'_spread' 
# for year in np.arange(2017,2023,1):
for year in np.arange(2005,2017,1):
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"
    date_range = pd.date_range(start=start_date, end=end_date)
    # files = [os.path.join(VERIFDIR,'T2m',f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
    if 2017 <= year <=2023:
        VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_v2p0_reforecast'
    if 2011 <= year <=2016:
        VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_v2p0_hindcast_fold_10'
    if 2005 <= year <=2010:
        VERIFDIR = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/CPC/Images_v2p0_hindcast_fold_9'
    
    files = [os.path.join(VERIFDIR, date, varname,f'{varname}.{date}.nc') for date in date_range.strftime('%Y%m%d')]
    ds = xr.open_mfdataset(files)
    ds = check_lat_order(ds,verbose=False)

    for label,lt in zip(['wk34'],[(21,28)]):
        # new dataset with current lead time. if more than one, concatenate lead times
        newds = xr.concat([ds.sel(lead_time=f'{i} days') for i in lt],dim='lead_time').mean('lead_time')
        anom   = newds[anomvar]
        spread = newds[spreadvar]



    # ds_obs = xr.open_dataset(f'/Projects/jalbers_process/CPC_LIM/t2m_data_for_scoring/final_v2_11.27.2023/T2m.cpc.grid2.ForecastLead_34.{year}to{year}.nc')
    # ds_obs = check_lat_order(ds_obs,verbose=False)
    # obs = ds_obs['cpc_anom_week34']
    # anom_J = ds_obs['lim_week34']


    ds_jra = xr.open_dataset(f'/Projects/jalbers_process/CPC_LIM/t2m_data_for_scoring/final_v2_11.27.2023/T2m.jra55.grid2.ForecastLead_34.{year}to{year}.nc')
    obs_jra = ds_jra['cpc_anom_week34']
    obs_jra = obs_jra.rename({'lons': 'lon'})
    obs_jra = obs_jra.rename({'lats': 'lat'})
    obs_jra = obs_jra.assign_coords(lon=ds_jra.lon.data)
    obs_jra = obs_jra.assign_coords(lat=ds_jra.lat.data)

    anom_J = ds_jra['lim_week34']
    anom_J = anom_J.rename({'lons': 'lon'})
    anom_J = anom_J.rename({'lats': 'lat'})
    anom_J = anom_J.assign_coords(lon=ds_jra.lon.data)
    anom_J = anom_J.assign_coords(lat=ds_jra.lat.data)

    # # mask    = xr.where(obs_jra.isnull() | anom_J.isnull() | spread.isnull() ,np.nan  ,1.)
    # # obs_jra = obs_jra.where(~mask.isnull(), drop=True)
    # # anom_J  = anom_J.where(~mask.isnull(), drop=True)
    # # spread  = spread.where(~mask.isnull(), drop=True)
    skill_HSS = []
    probability = []
    observation  = []
    for i, date in enumerate(date_range):
        warnings.filterwarnings("ignore")
        print(date)
        vCPC = obs_jra.isel(time=i)
        ANOM = anom_J.isel(time=i)
        SPREAD = spread.isel(time=i)
        mask   = xr.where(ANOM.isnull() | vCPC.isnull() | SPREAD.isnull() ,np.nan  ,1.)


        # Apply the mask to the datasets
        vCPC = vCPC.where(~mask.isnull(), drop=True)
        ANOM = ANOM.where(~mask.isnull(), drop=True)
        SPREAD = SPREAD.where(~mask.isnull(), drop=True)
    
        vCPC   = np.array(vCPC)[vCPC.notnull()]
        ANOM   = np.array(ANOM)[ANOM.notnull()]
        SPREAD = np.array(SPREAD)[SPREAD.notnull()]

        bounds = [-np.inf*np.ones(len(vCPC)),np.zeros(len(vCPC)),np.inf*np.ones(len(vCPC))]
        OBS = get_categorical_obs((vCPC,),bounds)[0]

        bounds = [-np.inf*np.ones(len(ANOM)),np.zeros(len(ANOM)),np.inf*np.ones(len(ANOM))]
        PROB = get_categorical_fcst((ANOM,),(SPREAD,),bounds)[0]

        HSS = get_heidke(PROB.T,OBS.T,categorical=True)

        skill_HSS.append(HSS)
        probability.append(PROB)
        observation.append(OBS)



    ds = xr.Dataset(
        {
            'observation':(['time','event','point'],np.array(observation)),
            'probability':(['time','event','point'],np.array(probability)),
            'HSS':(['time'],np.array(skill_HSS)),
        },
        coords={
            'time': date_range,
            'event': ('event',np.array([0,1]),{'long_name': 'cold and warm event probability'}),
            'point': np.arange(np.array(probability).shape[2]),
        }
    )


    SCORDIR = f'{VERIFDIR}/verification'
    fout = f'{SCORDIR}/{year}.nc'
    os.system(f'mkdir -p {SCORDIR}')
    os.system(f'rm -f {fout}')
    ds.to_netcdf(fout)
# Load .mat file
# mat_data = scipy.io.loadmat('/Projects/jalbers_process/CPC_LIM/t2m_data_for_scoring/scoring_data/final_v2_11.27.2023/jra55/scoring_data_2019.mat')
# mat_data['HSS_LIM'].mean()




