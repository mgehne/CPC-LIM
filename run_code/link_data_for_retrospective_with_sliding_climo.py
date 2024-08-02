import os
import numpy as np
from datetime import datetime as dt,timedelta
import netCDF4 as nc
from lib.tools import save_ncds
import glob

expt_number = '15'
# var_choice = 'Sam_vars'
# var_choice = '8_vars'
# var_choice = 'seasonally_varying_vars'
var_choice = 'seasonally_varying_vars_H500_truncation_test'
# forecast_mode = 'reforecast'
forecast_mode = 'hindcast_fold_10'
# forecast_mode = 'hindcast_fold_9'

expt_name = f'{expt_number}_{var_choice}_{forecast_mode}'

training_periods = {
"reforecast"     : {"period1": (1958,2016)},
"hindcast_fold_10": {"period1": (1958,2010)},
"hindcast_fold_9" : {"period1": (1958,2004), "period2":(2011,2016)},
                               
}

forecast_periods = {
"reforecast"     : {"period": (2017,2022)},
"hindcast_fold_10": {"period": (2011,2016)},
"hindcast_fold_9" : {"period": (2005,2010)},    
}

in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b2_sliding_climo_no_double_running_mean"
out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}"

if var_choice == '8_vars':
    varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
elif var_choice == 'seasonally_varying_vars':
    varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
elif var_choice == 'seasonally_varying_vars_H500_truncation_test':
    varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
elif var_choice == '7_vars':
    varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100"]
elif var_choice == 'Sam_vars':
    varnames = ["T2m", "SLP", "colIrr", "H500", "SF100"]


for period, years in training_periods[forecast_mode].items():
    start_year, end_year = years
    print(varnames) 
    print(period,f'{start_year} -- {end_year}')

    for year in range(start_year, end_year + 1):
        for varname in varnames:
            print(f"---------------- linking {year} for {varname} now ----------------")
            os.makedirs(os.path.join(out_data_folder, varname), exist_ok=True)
            source_file = os.path.join(in_data_folder, str(year), varname, f"{varname}.{year}.nc")
            target_file = os.path.join(out_data_folder, varname)
            
            if os.path.exists(source_file):
                os.symlink(source_file, os.path.join(target_file, f"{varname}.{year}.nc"))
            else:
                print(f"!!!missing {source_file} !!!!!")



##### Now make the ICs and save to data_retrospective #####
print("Now link files, make the ICs, and save to data_retrospective")
out_data_folder_retrospective=f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective'
os.system(f'mkdir -p {out_data_folder_retrospective}')

def concatenate_yearly_files(file_list, variable_name, output_filename):

    '''
    Accepts( a list of files, a Variable name, an output file)

    returns one huge combined list of files
    '''

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

    concatenate_yearly_files.__doc__

        

for varname in varnames:
    if os.path.exists(os.path.join(out_data_folder_retrospective,varname)):
        os.system(f'rm -rf {os.path.join(out_data_folder_retrospective,varname)}')
    os.system(f'mkdir {out_data_folder_retrospective}/{varname}')

    for period, years in forecast_periods[forecast_mode].items():
        start_year, end_year = years
        print(start_year,'--',end_year,f'for {varname}')
        for year in range(start_year,end_year+1):
            source_file = os.path.join(in_data_folder, str(year), varname, f"{varname}.{year}.nc")
            target_file = os.path.join(out_data_folder_retrospective, varname)
            
            if os.path.exists(source_file):
                os.symlink(source_file, os.path.join(target_file, f"{varname}.{year}.nc"))

    print(f'--------- processing {varname} ---------')
    filenames = sorted(glob.glob(f'{out_data_folder_retrospective}/{varname}/{varname}*nc'))
    print(filenames)
    
    try:
        os.system(f'rm {out_data_folder_retrospective}/{varname}_All.nc')
    except OSError:
        pass

    concatenate_yearly_files(filenames, 'anomaly', f'{out_data_folder_retrospective}/{varname}_All.nc')