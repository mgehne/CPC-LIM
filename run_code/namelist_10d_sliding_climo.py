r"""
Namelist for use in training and running a LIM

Sam Lillo, Matt Newman, John Albers
"""
# from run_for_realtime_10d_update_file_IO import FORECASTDAYS
# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================
# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
# climoyears = (1991,2020)
climoyears = (1996,2015)# This would be the climo for 2016, which is the last file read in and hence the climo is based on
expt_name = '10d_sliding_climo' # This experiment uses the same EOFs copied from 9_sliding_climo.   

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
# Need to mkdir of '*/data_clim/tmp'
VAR_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_clim/tmp/fullyr_JRA_58-16_sliding_climo_'
EOF_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/data_clim/tmp/EOF_JRA_58-16_sliding_climo_'
SLIDING_CLIMO_FILE_PREFIX = 'data_clim/'

# VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_58-14_climo_95-14'
# EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_58-14_climo_95-14'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
# RT_VARS = { 'SOIL':  {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SOIL/SOIL.2023.nc',
# 					'varname':'anomaly'},
# 			'SF100': {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SF100/SF100.2023.nc',
# 					'varname':'anomaly'},
# 			'SF750': {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SF750/SF750.2023.nc',
# 					'varname':'anomaly'},
# 			'H500':  {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/H500/H500.2023.nc',
# 					'varname':'anomaly'},
# 			'SLP':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SLP/SLP.2023.nc',
# 					'varname':'anomaly'},
# 			'T2m':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/T2m/T2m.2023.nc',
# 					'varname':'anomaly'},
# 			'colIrr':{'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/colIrr/colIrr.2023.nc',
# 					'varname':'anomaly'},
#      		'SST':   {'filename':f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/{expt_name}/SST/SST.2023.nc',
# 					'varname':'anomaly'}
#      }
# RT_VARS = {
# 			'H100':{'filename':'data_realtime/hgtAll.nc',
# 					'varname':'gh','level':100,'levname':'isobaricInhPa'},
# 			'H500':{'filename':'data_realtime/hgtAll.nc',
# 					'varname':'gh','level':500,'levname':'isobaricInhPa'},
# 			'SLP':{'filename':'data_realtime/surfAll.nc',
# 					'varname':'msl'},
# 			'T2m': {'info':('data_realtime/surfAll.nc','t2m',
#                                         {'latbounds':(20,74),
#                                          'lonbounds':(190,305),
#                                         'datebounds':datebounds,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2,
#                                         'season0':False,
#                                         'landmask':True})},
# 			'colIrr':{'filename':'data_realtime/phy2mAll.nc',
# 					'varname':'colIrr'},
			# }
# RT_VARS = {
# 			'H100':{'filename':'data_realtime/hgtAll.nc',
# 					'varname':'gh','level':100,'levname':'isobaricInhPa'},
# 			'H500':{'filename':'data_realtime/hgtAll.nc',
# 					'varname':'gh','level':500,'levname':'isobaricInhPa'},
# 			'SLP':{'filename':'data_realtime/surfAll.nc',
# 					'varname':'msl'},
# 			'T2m': {'filename':'data_realtime/surfAll.nc',
#                     'varname':'t2m'},
# 			'colIrr':{'filename':'data_realtime/phy2mAll.nc',
# 					'varname':'colIrr'},
# 			}
RT_VARS = { 'SOIL':{'filename':'/data/ycheng/JRA/Data/2023_realtime/land_2023.nc',
					'varname':'ussl'},
			'SF100':{'filename':'/data/ycheng/JRA/Data/2023_realtime/sf_2023.nc',
					'varname':'strf','level':100,'levname':'isobaricInhPa'},
			'SF750':{'filename':'/data/ycheng/JRA/Data/2023_realtime/sf_2023.nc',
					'varname':'strf','level':750,'levname':'isobaricInhPa'},
			'H500':{'filename':'/data/ycheng/JRA/Data/2023_realtime/hgt_2023.nc',
					'varname':'gh','level':500,'levname':'isobaricInhPa'},
			'SLP':{'filename':'/data/ycheng/JRA/Data/2023_realtime/surf_2023.nc',
					'varname':'msl'},
			'T2m':{'filename':'/data/ycheng/JRA/Data/2023_realtime/surf_2023.nc',
					'varname':'t2m'},
			'colIrr':{'filename':'/data/ycheng/JRA/Data/2023_realtime/phy2m_2023.nc',
					'varname':'colIrr'},
     		'SST':{'filename':'/data/ycheng/JRA/Data/2023_realtime/sst_2023.nc',
					'varname':'btmp'}
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
                                        'coarsegrain':2,
                                        'landmask':True})},
    
            'SST':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/SF750','anomaly',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SF100':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/SF100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'T2m':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'H500':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/H500','anomaly',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'colIrr':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SOIL':
                {'info':('/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo/SOIL','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
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
