# import xarray as xr
import os
from os import system
import glob
import numpy as np
from datetime import datetime as dt,timedelta
import netCDF4 as nc
from lib.tools import save_ncds
import copy

sliding_climo=True
subsetting_data = False
# subsetting_data = True
# in_data_path  = '/data/ycheng/JRA/Data'
# out_data_path = '/scratch/ycheng/JRA/Data/data_retrospective/expected_skills'
in_data_path  = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo'
out_data_path = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/data_retrospective'
os.system(f'mkdir -p {out_data_path}')

def concatenate_yearly_files(file_list, variable_name, output_filename):

    # Loop through the remaining files and concatenate along the time dimension
    ds = {}
    attrs = {}
    for prog, file in enumerate(file_list):
        ds0 = nc.Dataset(file)
        lat_name = ([s for s in ds0.variables.keys() if 'lat' in s]+[None])[0]
        lon_name = ([s for s in ds0.variables.keys() if 'lon' in s]+[None])[0]
        time_name = ([s for s in ds0.variables.keys() if 'time' in s]+[None])[0]
        var_name = variable_name

        try:
            attrs['long_name']=ds0[var_name].long_name
        except:
            attrs['long_name']=None
        try:
            attrs['units']=ds0[var_name].units
        except:
            attrs['units']=None
        
        try:
            timeunits = ds0[time_name].Units
        except:
            timeunits = ds0[time_name].units
        
        tmp = nc.num2date(ds0[time_name][:],timeunits,\
                only_use_cftime_datetimes=False,only_use_python_datetimes=True)
        if prog==0:
            ds['time'] = tmp
        else:
            ds['time'] = np.append(ds['time'],tmp)
        
        ds['lat']=ds0[lat_name][:]
        ds['lon']=ds0[lon_name][:]%360    
        newdata = ds0[var_name][:]
        if prog==0:
            ds['var'] = newdata
        else:
            ds['var'] = np.append(ds['var'],newdata,axis=0)

    try:
        attrs = copy.copy(attrs)
    except:
        attrs = {}

    vardict = {"anomaly": {'dims':("time","lat","lon"),
    # vardict = {f'{variable_name}': {'dims':("time","lat","lon"),
                            'data':ds['var'],
                            'attrs':attrs},
                }
    coords={
        "lon": {'dims':('lon',),'data':ds['lon'],
                'attrs':{'long_name':'longitude','units':'degrees_east'}},
        "lat": {'dims':('lat',),'data':ds['lat'],
                'attrs':{'long_name':'latitude','units':'degrees_north'}},
        "time": {'dims':('time',),'data':ds['time'],
                    'attrs':{'long_name':'time'}},

    }
    save_ncds(vardict,coords,filename=output_filename)

        
RT_VARS = ['T2m','SOIL','SLP','colIrr','H500','SST','SF100','SF750']
# RT_VARS = ['T2m'] 
years   = dt
for var in RT_VARS:
    os.system(f'rm -r {out_data_path}/{var}')
    os.system(f'mkdir {out_data_path}/{var}')
    # for year in range(2016,2023,1):
    for year in range(2017,2023,1):
    # for year in range(1995,2015,1):
        if sliding_climo:
            os.system(f'ln -s {in_data_path}/{year}/{var}/{var}.{year}.nc {out_data_path}/{var}')

        else:
            if var=='hgt' or var=='sf':
                os.system(f'ln -s {in_data_path}/{year}/{var}_{year}_1p25.nc {out_data_path}/{var}')
            else:
                os.system(f'ln -s {in_data_path}/{year}/{var}_{year}.nc {out_data_path}/{var}')

    print(f'--------- processing {var} ---------')
    filenames = sorted(glob.glob(f'{out_data_path}/{var}/{var}*nc'))
    print(filenames)
    # ds = xr.open_mfdataset(filenames, combine='nested', concat_dim='time')
    
    # if subsetting_data:
    #     T_START = dt(2016,12,20) #dt(YEAR,MONTH,1) 
    #     T_END = dt(2022,12,31) #dt(YEAR,MONTH,LASTDAY)
    #     selected_dates = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]
    #     ds = ds.sel(time=selected_dates)
    # Initialize the output dataset
        
    # print(ds['time'][0:10],ds['time'][-10::])
    try:
        os.system(f'rm {out_data_path}/{var}_All.nc')
    except OSError:
        pass

    concatenate_yearly_files(filenames, 'anomaly', f'{out_data_path}/{var}_All.nc')
    # concatenate_yearly_files(filenames, f'{var}', f'{out_data_path}/{var}_All.nc')

    