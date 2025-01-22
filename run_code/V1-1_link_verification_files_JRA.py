import os
# import numpy as np
# from datetime import datetime as dt,timedelta
# import netCDF4 as nc
# from lib.tools import save_ncds
# import glob

#### V1-1 and V1-2 only needs to be run if you want to create a new validation dataset.
# For example, if you want to verify using the sliding climo, then you only need to create v2p0 for the first time.
# Afterwards, even if you have different expt setup but you probably still want to verify against the same v2p0 validation.

# expt_number = 'v2p0'
# expt_number = 'v2p0_test'
expt_number = 'fixed_58-16_climo'

# CPC = True
CPC = False
if CPC:
    in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology_cpc/data"
    out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/CPC_verification"
    varnames = ["tavg"]
    full_years = list(range(1979, 2024))

else:
    if expt_number == 'v2p0':
        in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b2_sliding_climo_no_double_running_mean"
    elif expt_number == 'fixed_58-16_climo':
        in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/fixed_58-16_climo"
    out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_number}_verification"
    # varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
    varnames = ["T2m"]
    full_years = list(range(1958, 2024))

os.makedirs(out_data_folder, exist_ok=True)

print(f'varnames = {varnames}') 
print("Now link files for the verification period")
# copy the mask file
source_file = os.path.join(in_data_folder,f"mask.nc")
os.system(f'cp -r {source_file} {out_data_folder}')
#####

for year in full_years:
    for varname in varnames:
        # print(f"---------------- linking {year} for {varname} now ----------------")
        if CPC:
            os.makedirs(os.path.join(out_data_folder), exist_ok=True)
            source_file = os.path.join(in_data_folder,f"{varname}.{year}.2p0.nc")
        else:
            os.makedirs(os.path.join(out_data_folder, varname), exist_ok=True)
            if expt_number == 'v2p0':
                source_file = os.path.join(in_data_folder, str(year), varname,f"{varname}.{year}.nc")
            elif expt_number == 'fixed_58-16_climo':        
                source_file = os.path.join(in_data_folder, varname,f"{varname}.{year}.nc")
        target_file = os.path.join(out_data_folder, varname, 'links')
        os.makedirs(target_file, exist_ok=True)
        
        if os.path.exists(source_file):
            os.system(f'ln -sf {source_file} {os.path.join(target_file, f"{varname}.{year}.nc")}')
        else:
            print(f"!!!missing {source_file} !!!!!")
        ## Link mask.nc

