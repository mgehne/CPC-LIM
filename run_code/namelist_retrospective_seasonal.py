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
climoyears = (1991,2020)

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_58-14_climo_91-20'
EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_58-14_climo_91-20_seasonal'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 'SOIL':{'filename':'data_retrospective/landAll.nc',
					'varname':'ussl'},
			'SF100':{'filename':'data_retrospective/sfAll.nc',
					'varname':'strf','level':100,'levname':'isobaricInhPa'},
			'SF750':{'filename':'data_retrospective/sfAll.nc',
					'varname':'strf','level':750,'levname':'isobaricInhPa'},
			'H500':{'filename':'data_retrospective/hgtAll.nc',
					'varname':'gh','level':500,'levname':'isobaricInhPa'},
			'SLP':{'filename':'data_retrospective/surfAll.nc',
					'varname':'msl'},
			'T2m':{'filename':'data_retrospective/surfAll.nc',
					'varname':'t2m'},
			'colIrr':{'filename':'data_retrospective/phy2mAll.nc',
					'varname':'colIrr'},
     		'SST':{'filename':'data_retrospective/sstAll.nc',
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
                {'info':('/data/ycheng/JRA/Data/91-20_climo/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/SF750','anomaly',
                                        {'level':750,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'SF100':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/SF100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'T2m':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'H500':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/H500','anomaly',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'colIrr':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False})},
            'SOIL':
                {'info':('/data/ycheng/JRA/Data/91-20_climo/SOIL','anomaly',
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

        


# eof_trunc = {
#             mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
#             }
# eof_trunc_reg = {
#             mn: {'colIrr':23,'H500':8,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8,'CPCtemp':5} for mn in range(1,13)
#             }

eof_trunc = {
            mn: {'T2m':7,'SOIL':5} for mn in range(1,5)
            }
eof_trunc_reg = {
            mn: {'T2m':7,'SOIL':5} for mn in range(1,5)
            }
