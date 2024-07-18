r"""
Namelist for use in training and running a LIM

Sam Lillo, Yuan-Ming Cheng, John Albers, Maria Gehne, and Matt Newman
"""

import os

# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
climoyears = (1958, 2016)# This should be the last yearly file read in, needs to manually change.
use_expt_name_data = 'fixed_58-16_climo_reforecast'

# Variable, EOF pickle files prefix and add_offset files for sliding climo
retrospective_data_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data'
expt_path= os.path.join(retrospective_data_path,use_expt_name_data)
expt_data_clim_path = os.path.join(retrospective_data_path,use_expt_name_data,"data_clim")

os.system(f'mkdir -p {expt_data_clim_path}')
os.system(f'mkdir -p {expt_data_clim_path}/tmp')

VAR_FILE_PREFIX = f'{expt_data_clim_path}/tmp/fullyr_JRA_58-16_sliding_climo_' # These should be mannually changed based on the training period
EOF_FILE_PREFIX = f'{expt_data_clim_path}/tmp/EOF_JRA_58-16_sliding_climo_'
SLIDING_CLIMO_FILE_PREFIX = expt_data_clim_path


# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 

			'SF100': {'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/SF100_All.nc',
					'varname':'anomaly'},
			'SF750': {'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/SF750_All.nc',
					'varname':'anomaly'},		
			'H500':  {'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/H500_All.nc',
					'varname':'anomaly'},
			'SLP':   {'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/SLP_All.nc',
					'varname':'anomaly'},
			'T2m':   {'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/T2m_All.nc',
					'varname':'anomaly'},
			'colIrr':{'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/colIrr_All.nc',
					'varname':'anomaly'},
   			'SOIL'  :{'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/SOIL_All.nc',
					'varname':'anomaly'},
			'SST'   :{'filename':f'{retrospective_data_path}/{use_expt_name_data}/data_retrospective/SST_All.nc',
					'varname':'anomaly'},

     }

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
use_vars = {


            'SF100':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/SF100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SF750':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/SF750','anomaly',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'T2m':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'H500':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/H500','anomaly',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'colIrr':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SST':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SOIL':
                {'info':(f'{retrospective_data_path}/{use_expt_name_data}/SOIL','anomaly',
                                        {'latbounds':(24,74),# This is the (only and) major change
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
                }    

''' 
Set EOF truncations for variables.
Dictionary keys refer to the variable (or variables within a tuple for a combined EOF).
Dictionary values refer to the respective EOF truncation.
Keys in eof_trunc dictionary refer to month of the year.
'''

         



eof_trunc = {
            mn:{'colIrr':23,'H500':14,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            }
eof_trunc_reg = {
            mn:{'colIrr':23,'H500':14,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)            
            }

