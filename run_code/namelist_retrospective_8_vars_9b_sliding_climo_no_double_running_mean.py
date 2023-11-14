r"""
Namelist for use in training and running a LIM

Sam Lillo, Matt Newman, John Albers
"""

# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
# climoyears = (1991,2020)
climoyears = (1996,2015)# This would be the climo for 2016, which is the last file read in and hence the climo is based on

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
# Need to mkdir of '*/data_clim/tmp'
VAR_FILE_PREFIX = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_clim/tmp/fullyr_JRA_58-16_sliding_climo_'
EOF_FILE_PREFIX = '/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_clim/tmp/EOF_JRA_58-16_sliding_climo_'
# VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_58-14_climo_95-14'
# EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_58-14_climo_95-14'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 'SOIL':  {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/SOIL_All.nc',
					'varname':'anomaly'},
			'SF100': {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/SF100_All.nc',
					'varname':'anomaly'},
			'SF750': {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/SF750_All.nc',
					'varname':'anomaly'},
			'H500':  {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/H500_All.nc',
					'varname':'anomaly'},
			'SLP':   {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/SLP_All.nc',
					'varname':'anomaly'},
			'T2m':   {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/T2m_All.nc',
					'varname':'anomaly'},
			'colIrr':{'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/colIrr_All.nc',
					'varname':'anomaly'},
     		'SST':   {'filename':'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/data_retrospective_subset/SST_All.nc',
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
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'landmask':True})},
    
            'SST':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/SF750','anomaly',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SF100':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/SF100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'T2m':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'H500':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/H500','anomaly',
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
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SOIL':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9b_sliding_climo_no_double_running_mean/SOIL','anomaly',
                                        # {'latbounds':(20,74),
                                        {'latbounds':(30,74),
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
            mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            }
eof_trunc_reg = {
            mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8,'CPCtemp':5} for mn in range(1,13)
            }

# eof_trunc = {
#             mn: {'T2m':7,'SOIL':5} for mn in range(1,13)
#             }
# eof_trunc_reg = {
#             mn: {'T2m':7,'SOIL':5} for mn in range(1,13)
#             }
