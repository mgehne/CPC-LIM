import os
# import numpy as np
# from datetime import datetime as dt,timedelta
# import netCDF4 as nc
# from lib.tools import save_ncds
# import glob

expt_number = 'v2p0'

CPC = True
# CPC = False
if CPC:
    in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology_cpc/data"
    out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/CPC_verification"
    varnames = ["tavg"]
    full_years = list(range(1979, 2024))

else:
    in_data_folder = "/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b2_sliding_climo_no_double_running_mean"
    out_data_folder = f"/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_number}_verification"
    varnames = ["T2m", "SOIL", "SLP", "colIrr", "H500", "SST", "SF100", "SF750"]
    full_years = list(range(1958, 2024))

os.makedirs(out_data_folder, exist_ok=True)

print(f'varnames = {varnames}') 
print("Now link files for the verification period")
for year in full_years:
    for varname in varnames:
        # print(f"---------------- linking {year} for {varname} now ----------------")
        if CPC:
            os.makedirs(os.path.join(out_data_folder), exist_ok=True)
            source_file = os.path.join(in_data_folder,f"{varname}.{year}.2p0.nc")
        else:
            os.makedirs(os.path.join(out_data_folder, varname), exist_ok=True)
            source_file = os.path.join(in_data_folder, str(year), varname,f"{varname}.{year}.nc")
        target_file = os.path.join(out_data_folder, varname, 'links')
        os.makedirs(target_file, exist_ok=True)
        
        if os.path.exists(source_file):
            os.system(f'ln -sf {source_file} {os.path.join(target_file, f"{varname}.{year}.nc")}')
        else:
            print(f"!!!missing {source_file} !!!!!")