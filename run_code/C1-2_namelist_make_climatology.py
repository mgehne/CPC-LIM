r"""
Namelist for creating climatology using existing code structure.

Sam Lillo, Yuan-Ming Cheng, Matt Newman, John Albers
"""
import os  
# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
# climoyears = (1971,2000)
# climoyears = (1981,2010)
climoyears = (1991,2020)

resolution = 2

use_expt_name_data = f'{resolution}p0.{climoyears[0]}-{climoyears[1]}'

# This folder holds all 3 different periods of climatology
climatology_data_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology'

# This folder is for current climoyears
expt_path= os.path.join(climatology_data_path,use_expt_name_data) 

# This VAR_FILE_PREFIX determines where pickle files and nc files are dumped to and give the file a consistent prefix, e.g., 2p0.1991-2020_
VAR_FILE_PREFIX = f'{expt_path}/{resolution}p0.{climoyears[0]}-{climoyears[1]}_'# 

os.system(f'mkdir -p {climatology_data_path}')
os.system(f'mkdir -p {expt_path}') 

''' 
Set variables for calculating climatology. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the directory where the input data are in the 1st elements of 'info'. This is the data where the climatology will be based on  
The rest of 'info' contains all the flags and arguments for processing the input data. 
'''
use_vars = {
            # 'CPCtemp':
            #     {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/cpc_offline_climatology/{resolution}p0.{climoyears[0]}-{climoyears[1]}/CPCtemp','anomaly',
            #                             {'latbounds':(20,74),
            #                              'lonbounds':(190,305),
            #                             'datebounds':datebounds,
            #                             'season0':False,#  climo would use climoyears as bounds for years
            #                             'climoyears':climoyears,
            #                             'time_window':time_window,
            #                             'coarsegrain':resolution,
            #                             'landmask':True})},

            'T2m':
                {'info':(f'/data/ycheng/JRA/Data/make_rawdata_climatology/{resolution}p0.{climoyears[0]}-{climoyears[1]}/surf','t2m',
                                        {'latbounds':(20,74),
                                        'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window, 
                                        # In this script, time_window and the running mean is only relevant for anomaly. We are calculating climo here. 
                                        # This flag doesnt change our climo.
                                        'coarsegrain':resolution,
                                        'landmask':True,})},
            
                }    


# %%
