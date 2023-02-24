r"""
Namelist for use in training and running a LIM
"""
# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
climoyears = (1981,2010)

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_79-17_'
EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_79-17_'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
    
use_vars = {
            'CPCtemp':
                {'info':('./data_clim/cpcdata_2p5','tavg',
                                        {'latbounds':(20,74),
                                        'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'season0':True,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True})},
}