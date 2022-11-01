#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:14:10 2021

@author: slillo
"""


# %%===========================================================================
# SET LIM AND DATA SPECIFICATIONS
# =============================================================================

# Set time window for averaging and tau, and date bounds for season.
time_window = 7
tau1n = 5
datebounds = ('1/1','12/31')
climoyears = (1979,2017)

runyear = 2017
runmonth = 1

# Variable and EOF object file prefix
VAR_FILE_PREFIX = 'data_clim/fullyr_JRA_79-17_'
EOF_FILE_PREFIX = 'data_clim/EOF_JRA_79-17_'

# Path for teleconnection loading patterns
TELECONNECTION_PATTERN_NCFILE = 'data_clim/teleconnection_loading_patterns.nc'
RMM_PATTERN_NCFILE = 'data_clim/RMM_loading_patterns.nc'

''' 
Set filenames and variable name within the file for realtime data.
Dictionary keys must match those in use_vars.
'''

RT_VARS = {
 			'H100':{'filename':f'data_hindcast_new/{runyear}{runmonth:02}/hgt.nc',
 					'varname':'gh','level':100,'levname':'isobaricInhPa'},
 			'H500':{'filename':f'data_hindcast_new/{runyear}{runmonth:02}/hgt.nc',
 					'varname':'gh','level':500,'levname':'isobaricInhPa'},
 			'SLP':{'filename':f'data_hindcast_new/{runyear}{runmonth:02}/prmsl.nc',
 					'varname':'msl'},
 			'T2m':{'filename':f'data_hindcast_new/{runyear}{runmonth:02}/tmp.nc',
 					'varname':'t2m'},
 			'colIrr':{'filename':f'data_hindcast_new/{runyear}{runmonth:02}/colIrr.nc',
 					'varname':'colIrr'},
 			}
# RT_VARS = {
#  			'H100':{'filename':'data_realtime/hgtAll.nc',
#  					'varname':'gh','level':100,'levname':'isobaricInhPa'},
#  			'H500':{'filename':'data_realtime/hgtAll.nc',
#  					'varname':'gh','level':500,'levname':'isobaricInhPa'},
#  			'SLP':{'filename':'data_realtime/surfAll.nc',
#  					'varname':'msl'},
#  			'T2m':{'filename':'data_realtime/surfAll.nc',
#  					'varname':'t2m'},
#  			'colIrr':{'filename':'data_realtime/phy2mAll.nc',
#  					'varname':'colIrr'},
#  			}

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
    
use_vars = {
            'H100':
                {'info':('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':100,
                                        'latbounds':(30,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'H500':
                {'info':('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/geopot/','geopot',
                                        {'level':500,
                                        'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'SLP':
                {'info':('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/mslp/','mslp',
                                        {'latbounds':(20,90),
                                        'lonbounds':(0,360),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':5})},
            'T2m':
                {'info':('/Volumes/time machine backup/ALL_LIM_STUFF/Data/JRA_t2m/','TMP_GDS0_HTGL',
                                        {'latbounds':(20,74),
                                         'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'season0':True,
                                        'climoyears':climoyears,
                                        'time_window':time_window,
                                        'coarsegrain':2.5,
                                        'landmask':True})},
            'colIrr':
                {'info':('/Volumes/time machine backup/ALL_LIM_STUFF/Data_res2/colirr/','colIrradiance',
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

#eof_trunc = {
#            'fullyr': {'colIrr':23,'H500':14,'H100':12,'SLP':23,'T2m':5} for mn in range(1,13)
#            }
