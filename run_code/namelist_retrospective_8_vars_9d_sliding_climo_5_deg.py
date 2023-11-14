r"""
Namelist for use in training and running a LIM

Sam Lillo, Matt Newman, John Albers
"""

# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================
import os

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
# climoyears = (1991,2020)
climoyears = (2006,2015)# This would be the climo for 2016, which is the last file read in and hence the climo is based on
expt_name = '9d_sliding_climo_5_deg'

# Variable and EOF object file prefix

LIMpage_path = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}'
os.system(f'mkdir -p {LIMpage_path}/data_clim/')
os.system(f'mkdir -p {LIMpage_path}/data_clim/tmp')
# os.system(f'mkdir -p {LIMpage_path}/data_clim/')
# os.system(f'mkdir -p {LIMpage_path}/data_clim/tmp')
VAR_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_clim/tmp/fullyr_JRA_58-16_{expt_name}_'
EOF_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_clim/tmp/EOF_JRA_58-16_{expt_name}_'
# VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_58-14_climo_95-14'
# EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_58-14_climo_95-14'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 'SOIL':  {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/SOILAll.nc',
					'varname':'anomaly'},
			'SF100': {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/SF100All.nc',
					'varname':'anomaly'},
			'SF750': {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/SF750All.nc',
					'varname':'anomaly'},
			'H500':  {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/H500All.nc',
					'varname':'anomaly'},
			'SLP':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/SLPAll.nc',
					'varname':'anomaly'},
			'T2m':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/T2mAll.nc',
					'varname':'anomaly'},
			'colIrr':{'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/colIrrAll.nc',
					'varname':'anomaly'},
     		'SST':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_retrospective/SSTAll.nc',
					'varname':'anomaly'}
     }

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
use_vars = {
            'CPCtemp':
                {'info':('./data_clim/cpcdata','temp',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'season0':False,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'landmask':True})},
    
            'SST':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SF750','anomaly',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False
                                        })},
            'SF100':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SF100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False
                                        })},
            'T2m':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False
                                        })},
            'H500':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/H500','anomaly',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False
                                        })},
            'colIrr':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
                                        'season0':False
                                        })},
            'SOIL':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SOIL','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        # 'coarsegrain':2,
                                        'coarsegrain':5,
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
            mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            }
eof_trunc_reg = {
            mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            # mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8,'CPCtemp':5} for mn in range(1,13)
            }

# eof_trunc = {
#             mn: {'T2m':7,'SOIL':5} for mn in range(1,13)
#             }
# eof_trunc_reg = {
#             mn: {'T2m':7,'SOIL':5} for mn in range(1,13)
#             }
