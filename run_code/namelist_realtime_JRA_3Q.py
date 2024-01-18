r"""
Namelist for use in training and running a LIM

Sam Lillo, Yuan-Ming Cheng, John Albers, Maria Gehne, and Matt Newman
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
expt_name = 'realtime' # This experiment uses the same EOFs copied from 9_sliding_climo.   

# Variable and EOF object file prefix
VAR_FILE_PREFIX = f'data_clim/tmp/fullyr_JRA_58-16_sliding_climo_'
EOF_FILE_PREFIX = f'data_clim/tmp/EOF_JRA_58-16_sliding_climo_'
SLIDING_CLIMO_FILE_PREFIX = 'data_clim'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''
 
RT_VARS = { 'SOIL':{'filename':'data_realtime/landAll.nc',
					'varname':'liqvsm'},
			'SF100':{'filename':'data_realtime/sfAll.nc',
					'varname':'strf','level':100,'levname':'isobaricInhPa'},
			'SF750':{'filename':'data_realtime/sfAll.nc',
					'varname':'strf','level':750,'levname':'isobaricInhPa'},
			'H500':{'filename':'data_realtime/hgtAll.nc',
					'varname':'gh','level':500,'levname':'isobaricInhPa'},
			'SLP':{'filename':'data_realtime/surfAll.nc',
					'varname':'prmsl'},
			'T2m':{'filename':'data_realtime/surfAll.nc',
					'varname':'t2m'},
			'colIrr':{'filename':'data_realtime/phy2mAll.nc',
					'varname':'colIrr'},
     		'SST':{'filename':'data_realtime/sstAll.nc',
					'varname':'sst'}
     }
''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
use_vars = {
            # 'CPCtemp':
            #     {'info':('./data_clim/cpcdata','temp',
            #                             {'latbounds':(20,74),
            #                              'lonbounds':(190,305),
            #                             'datebounds':datebounds,
            #                             'season0':False,
            #                             'climoyears':climoyears,
            #                             # 'time_window':time_window,
            #                             'coarsegrain':2,
            #                             'landmask':True})},
    
            'SST':
                {'info':(f'rawdata/SST','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'oceanmask':True})},
            'SF750':
                {'info':(f'rawdata/SF750','anomaly',
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
                {'info':(f'rawdata/SF100','anomaly',
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
                {'info':(f'rawdata/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False,
                                        'landmask':True})},
            'SLP':
                {'info':(f'rawdata/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'H500':
                {'info':(f'rawdata/H500','anomaly',
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
                {'info':(f'rawdata/colIrr','anomaly',
                                        {'latbounds':(-14,14),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window,
                                        'coarsegrain':2,
                                        'season0':False
                                        })},
            'SOIL':
                {'info':(f'rawdata/SOIL','anomaly',
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
            mn: {'colIrr':23,'H500':14,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            }
eof_trunc_reg = {
            mn: {'colIrr':23,'H500':14,'SLP':20,'T2m':7,'SOIL':5,'SF750':8,'SF100':8,'SST':8} for mn in range(1,13)
            }

