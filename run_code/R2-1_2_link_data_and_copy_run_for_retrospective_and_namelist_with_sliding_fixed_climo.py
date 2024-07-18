import os
import numpy as np
from datetime import datetime as dt,timedelta
import netCDF4 as nc
from lib.tools import save_ncds
import glob

# expt_number = 'v2p0'
expt_number="fixed_58-16_climo"

forecast_mode = 'reforecast' #This may give error when modifying namelist and run files
# forecast_mode = 'hindcast_fold_10' This may give error when changing the namelist and run files
# forecast_mode = 'hindcast_fold_9'
# forecast_mode = 'hindcast_fold_8'
# forecast_mode = 'hindcast_fold_7'
# forecast_mode = 'hindcast_fold_6'
# forecast_mode = 'hindcast_fold_5'
# forecast_mode = 'hindcast_fold_4'
# forecast_mode = 'hindcast_fold_3'
# forecast_mode = 'hindcast_fold_2'
# forecast_mode = 'hindcast_fold_1'

sliding_climo=False

expt_name = f'{expt_number}_{forecast_mode}'
# expt_name = f'{expt_number}_{var_choice}_{forecast_mode}'

forecast_periods_input = {
"reforecast"     :  (2017,2022),
"hindcast_fold_10": (2011,2016),
"hindcast_fold_9" : (2005,2010),    
"hindcast_fold_8" : (1999,2004),    
"hindcast_fold_7" : (1993,1998),    
"hindcast_fold_6" : (1987,1992),    
"hindcast_fold_5" : (1981,1986),    
"hindcast_fold_4" : (1975,1980),    
"hindcast_fold_3" : (1969,1974),    
"hindcast_fold_2" : (1963,1968),    
"hindcast_fold_1" : (1958,1962),    
}

if forecast_mode == 'reforecast':
    full_years = list(range(1958,2023))
else:
    full_years = list(range(1958, 2017))

# create a list of forecast years
forecast_periods = np.arange(forecast_periods_input[forecast_mode][0],forecast_periods_input[forecast_mode][1]+1,1).tolist()

# create a list of training years
training_periods = [year for year in full_years if year < forecast_periods[0] or \
                                                   year > forecast_periods[len(forecast_periods)-1]]


print(f'------{forecast_mode}-----')
print(f'training years: {training_periods}')
print(f'forecast years: {forecast_periods}')

# in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b2_sliding_climo_no_double_running_mean"
in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/fixed_58-16_climo"
out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}"

varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]


print(f'varnames = {varnames}') 
print("Now link files for the training period")
for year in training_periods:
    for varname in varnames:
        # print(f"---------------- linking {year} for {varname} now ----------------")
        os.makedirs(os.path.join(out_data_folder, varname), exist_ok=True)
        if sliding_climo:
            source_file = os.path.join(in_data_folder, str(year), varname, f"{varname}.{year}.nc")
        else: 
            source_file = os.path.join(in_data_folder, varname, f"{varname}.{year}.nc")
        target_file = os.path.join(out_data_folder, varname)
        print(source_file)
        
        if os.path.exists(source_file):
            os.system(f'ln -sf {source_file} {os.path.join(target_file, f"{varname}.{year}.nc")}')
        else:
            print(f"!!!missing {source_file} !!!!!")



##### Now make the ICs and save to data_retrospective #####
print("Now link files, make the ICs, and save to data_retrospective")
out_data_folder_retrospective=f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective'
os.system(f'mkdir -p {out_data_folder_retrospective}')

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

        

for varname in varnames:
    if os.path.exists(os.path.join(out_data_folder_retrospective,varname)):
        os.system(f'rm -rf {os.path.join(out_data_folder_retrospective,varname)}')
    os.system(f'mkdir {out_data_folder_retrospective}/{varname}')

    for year in forecast_periods:
        if sliding_climo:
            source_file = os.path.join(in_data_folder, str(year), varname, f"{varname}.{year}.nc")
        else: 
            source_file = os.path.join(in_data_folder, varname, f"{varname}.{year}.nc")
        target_file = os.path.join(out_data_folder_retrospective, varname)
        
        if os.path.exists(source_file):
            os.symlink(source_file, os.path.join(target_file, f"{varname}.{year}.nc"))

    print(f'--------- processing {varname} ---------')
    filenames = sorted(glob.glob(f'{out_data_folder_retrospective}/{varname}/{varname}*nc'))
    print(f'Years in the initial conditions in data_retrospective: {forecast_periods}')
    # print(filenames)
    
    try:
        os.system(f'rm {out_data_folder_retrospective}/{varname}_All.nc')
    except OSError:
        pass

    concatenate_yearly_files(filenames, 'anomaly', f'{out_data_folder_retrospective}/{varname}_All.nc')

print("Now link climo offset files for years in forecast_periods")
for varname in varnames:
    for year in forecast_periods:
        # print(f"---------------- linking {year} for {varname} now ----------------")
        os.makedirs(os.path.join(out_data_folder, 'data_clim',varname), exist_ok=True)
        if sliding_climo:
            source_file = os.path.join(in_data_folder, str(year), varname, f"{varname}.{year}.nc")
        else:
            source_file = os.path.join(in_data_folder, varname, f"{varname}.{year}.nc")
        target_file = os.path.join(out_data_folder, 'data_clim',varname)
        
        if os.path.exists(source_file):
            # os.symlink(source_file, os.path.join(target_file, f"{varname}.{year}.nc"))
            os.system(f'ln -sf {source_file} {os.path.join(target_file, f"{varname}.{year}.nc")}')
        else:
            print(f"!!!missing {source_file} !!!!!")
            
    print(f'Years in {varname} climo offset files in data_clim: {forecast_periods}')


def copy_and_modify_namelist_v2p0_hindcast_fold_8(input_file, output_file, climoyears , use_expt_name_data, traing_period_string):
    """
    Copy a Python script and modify the years value.

    Parameters:
        input_file (str): Path to the input Python script.
        output_file (str): Path to the output Python script.
        climoyears (tuple): climoyears to be replaced.
        use_expt_name_date (str): use__expt_name_data to be replaced.
        traing_period_string (str): traing_period_string to be replaced.
    """
    os.system(f'rm -f {output_file}')
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            if "climoyears = (1996,2015)" in line:
                string_old =  '(1996,2015)'
                string_new = f'{climoyears}'
                line = line.replace(string_old, string_new)

            if "use_expt_name_data = 'v2p0_hindcast_fold_8'" in line:
                string_old = "v2p0_hindcast_fold_8"
                string_new = f"{use_expt_name_data}"
                line = line.replace(string_old, string_new)
                
            if "VAR_FILE_PREFIX" in line:
                string_old =  'fullyr_JRA_58-98_05-16_sliding_climo_' 
                string_new = f'fullyr_JRA_{traing_period_string}_sliding_climo_'
                line = line.replace(string_old, string_new)

            if "EOF_FILE_PREFIX" in line:
                string_old =  'EOF_JRA_58-98_05-16_sliding_climo_' 
                string_new = f'EOF_JRA_{traing_period_string}_sliding_climo_'
                line = line.replace(string_old, string_new)
            f.write(line)

def copy_and_modify_run_for_hindcast_fold_8(input_file, output_file, year_START, year_END):
    """
    Copy a Python script and modify the years value.

    Parameters:
        input_file (str): Path to the input Python script.
        output_file (str): Path to the output Python script.
        climoyears (tuple): climoyears to be replaced.
        use_expt_name_date (str): use__expt_name_data to be replaced.
        traing_period_string (str): traing_period_string to be replaced.
    """
    os.system(f'rm -f {output_file}')
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for line in lines:
            if "expt_name" in line:
                string_old = 'v2p0_hindcast_fold_8'
                string_new = f'{expt_name}'
                line = line.replace(string_old, string_new)

            if "T_START = dt(1999,1,1)" in line:
                string_old =  'dt(1999,1,1)'
                string_new = f'dt({year_START},1,1)'
                line = line.replace(string_old, string_new)

            if "T_END = dt(2004,12,31)" in line:
                string_old = "dt(2004,12,31)"
                string_new = f"dt({year_END},12,31)"
                line = line.replace(string_old, string_new)
            f.write(line)


# Copy and modify the script for each experiment
training_years_diff = [training_periods[i+1] - training_periods[i] for i in range(len(training_periods)-1)]
if len(set(training_years_diff)) == 1:
    print(f'All elements in training_years_diff are the same, {forecast_mode}')
    period_1_start = str(training_periods[0])[-2:]
    period_1_end   = str(training_periods[len(training_periods)-1])[-2:]
    period_2_start = ""
    period_2_end = ""
    traing_period_string = f'{period_1_start}-{period_1_end}'
else:
    # Take the maximum value from training_years_diff
    last_year_of_1st_period = training_years_diff.index(max(training_years_diff))
    period_1_start = str(training_periods[0])[-2:]
    period_1_end  = str(training_periods[last_year_of_1st_period])[-2:]
    period_2_start = str(training_periods[last_year_of_1st_period+1])[-2:]
    period_2_end  = str(training_periods[len(training_periods)-1])[-2:]
    traing_period_string = f'{period_1_start}-{period_1_end}_{period_2_start}-{period_2_end}'

# print(traing_period_string)
    
# for namelist
input_namelist = "namelist_v2p0_hindcast_fold_8.py"
output_namelist = f"namelist_{expt_name}.py"
if sliding_climo:
    climo_end = training_periods[len(training_periods)-1]-1
    climo_start = climo_end - 19
    climoyears = (climo_start,climo_end)
else: 
    climo_end = training_periods[len(training_periods)-1]
    climo_start = training_periods[0]
    climoyears = (climo_start,climo_end)
use_expt_name_data = f'{expt_name}'

copy_and_modify_namelist_v2p0_hindcast_fold_8(input_namelist, output_namelist, climoyears, use_expt_name_data, traing_period_string)

# for run_for_hindcast
year_START = forecast_periods[0]
year_END = forecast_periods[len(forecast_periods)-1]
input_namelist = "run_for_hindcast_fold_8.py"
output_namelist = f"run_for_{expt_name}.py"

copy_and_modify_run_for_hindcast_fold_8(input_namelist, output_namelist, year_START, year_END)
