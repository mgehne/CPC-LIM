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
climoyears = (1981,2020)

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_58-20_climo_91-20'
EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_58-20_climo_91-20'


# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 'SOIL':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/landAll.nc',
					'varname':'ussl'},
			'SF100':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/sfAll.nc',
					'varname':'strf','level':100,'levname':'isobaricInhPa'},
			'SF750':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/sfAll.nc',
					'varname':'strf','level':750,'levname':'isobaricInhPa'},
			'H500':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/hgtAll.nc',
					'varname':'gh','level':500,'levname':'isobaricInhPa'},
			'SLP':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/surfAll.nc',
					'varname':'msl'},
			'T2m':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/surfAll.nc',
					'varname':'t2m'},
			'colIrr':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/phy2mAll.nc',
					'varname':'colIrr'},
     		'SST':{'filename':'/scratch/ycheng/JRA/Data/data_retrospective/raw/sstAll.nc',
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
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/SF750','anomaly',
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
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/SF100','anomaly',
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
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'H500':
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/H500','anomaly',
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
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SOIL':
                {'info':('/scratch/ycheng/JRA/Data/91-20_climo_test_season0/SOIL','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
                }    
# use_vars = {
#              'CPCtemp':
#                 {'info':('./data_clim/cpcdata','temp',
#                                         {'latbounds':(20,74),
#                                          'lonbounds':(190,305),
#                                         'datebounds':datebounds,
#                                         'season0':True,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2,
#                                         'landmask':True})},
#             'SOIL':
#                 {'info':('./rawdata/SOIL','anomaly',
#                                         {'latbounds':(20,74),
#                                          'lonbounds':(190,305),
#                                         'datebounds':datebounds,
#                                         'season0':True,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2,
#                                         'landmask':True})},
#              'T2m':
#                 {'info':('./rawdata/T2m','anomaly',
#                                         {'latbounds':(20,74),
#                                          'lonbounds':(190,305),
#                                         'datebounds':datebounds,
#                                         'season0':True,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2,
#                                         'landmask':True})},
#             'H100':
#                 {'info':('./rawdata/H100','anomaly',
#                                         {'level':100,
#                                         'latbounds':(30,90),
#                                         'lonbounds':(0,360),
#                                         'datebounds':datebounds,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2})},
#             'H500':
#                 {'info':('./rawdata/H500','anomaly',
#                                         {'level':500,
#                                         'latbounds':(20,90),
#                                         'lonbounds':(0,360),
#                                         'datebounds':datebounds,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2})},
#             'SLP':
#                 {'info':('./rawdata/SLP','anomaly',
#                                         {'latbounds':(20,90),
#                                         'lonbounds':(0,360),
#                                         'datebounds':datebounds,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2})},
#             'colIrr':
#                 {'info':('./rawdata/colIrr','anomaly',
#                                         {'latbounds':(-20,20),
#                                          'lonbounds':(0,360),
#                                         'datebounds':datebounds,
#                                         'season0':True,
#                                         'climoyears':climoyears,
#                                         'time_window':time_window,
#                                         'coarsegrain':2})}
#             }
    

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
