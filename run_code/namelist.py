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
climoyears = (1979,2017)

# Variable and EOF object file prefix
#VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
VAR_FILE_PREFIX = 'data_clim/tmp/fullyr_JRA_79-17_'
EOF_FILE_PREFIX = 'data_clim/tmp/EOF_JRA_79-17_'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''

RT_VARS = {
			'H100':{'filename':'data_realtime/hgtAll.nc',
					'varname':'gh','level':100,'levname':'isobaricInhPa'},
			'H500':{'filename':'data_realtime/hgtAll.nc',
					'varname':'gh','level':500,'levname':'isobaricInhPa'},
			'SLP':{'filename':'data_realtime/surfAll.nc',
					'varname':'msl'},
			'T2m':{'filename':'data_realtime/surfAll.nc',
					'varname':'t2m'},
			'colIrr':{'filename':'data_realtime/phy2mAll.nc',
					'varname':'colIrr'},
			}

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
    
use_vars = {
             'CPCtemp':
                {'info':('./data_clim/cpcdata_hr','tavg',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                         'datebounds':datebounds,
                                         'season0':True,
                                         'climoyears':climoyears,
                                         'time_window':time_window,
                                         'coarsegrain':2.0})},
             'CPCtempHR':
                {'info':('./data_clim/cpcdata_hr','tavg',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                         'datebounds':datebounds,
                                         'season0':True,
                                         'climoyears':climoyears,
                                         'time_window':time_window,
                                         'landmask':True})},
             'T2m':
                {'info':('./rawdata/T2m','anomaly',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'season0':True,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True})},
            'H100':
                {'info':('./rawdata/H100','anomaly',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'H500':
                {'info':('./rawdata/H500','anomaly',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'SLP':
                {'info':('./rawdata/SLP','anomaly',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'colIrr':
                {'info':('./rawdata/colIrr','anomaly',
                                        {'latbounds':(-20,20),
                                         'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'season0':True,
                                        'climoyears':climoyears,
                                        'time_window':time_window,})}
            }
    

''' 
Set EOF truncations for variables.
Dictionary keys refer to the variable (or variables within a tuple for a combined EOF).
Dictionary values refer to the respective EOF truncation.
Keys in eof_trunc dictionary refer to month of the year.
'''


# eof_trunc = {
#             1: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             2: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             3: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             4: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             5: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             6: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             7: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             8: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             9: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             10: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             11: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             12: {'colIrr':10,'H500':14,('SF750','SF250'):15,('H10','H100'):12,'SLP':15,'T2m':5},
#             }

eof_trunc = {
            mn: {'colIrr':23,'H500':14,'H100':12,'SLP':23,'T2m':5} for mn in range(1,13)
            }
eof_trunc_reg = {
            mn: {'colIrr':23,'H500':14,'H100':12,'SLP':23,'T2m':5,'CPCtemp':5,'CPCtempHR':5} for mn in range(1,13)
            }            

#eof_trunc = {
#            'fullyr': {'colIrr':23,'H500':14,'H100':12,'SLP':23,'T2m':5} for mn in range(1,13)
#            }
