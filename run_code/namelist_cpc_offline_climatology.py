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
climoyears = (1981,2010)
# climoyears = (1991,2020)
# resolution = 5
resolution = 2
# Variable object file prefix
# Need to mkdir of '*/data_clim/tmp'
# VAR_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/cpc_offline_climatology/data_clim/tmp/{resolution}p0.CPC.{climoyears[0]}-{climoyears[1]}_'
# I need to change the CPC version to the new directory offline_climatology without 'cpc'
VAR_FILE_PREFIX = f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/offline_climatology/{resolution}p0.{climoyears[0]}-{climoyears[1]}/data_clim/{resolution}p0.{climoyears[0]}-{climoyears[1]}_'
# This directory needs to be mannualy made

''' 
Set variables to save for use in LIMs. 
Dictionary of variable names, each containing a dictionary with 
'info' and 'data'. Set the elements of 'info' here. 
For each variable, 'info' contains all the input arguments for the dataset. 
'''
use_vars = {
            # 'CPCtemp':
            #     {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/cpc_offline_climatology/{resolution}p0.{climoyears[0]}-{climoyears[1]}/CPCtemp','anomaly',
            #                             {'latbounds':(20,74),
            #                              'lonbounds':(190,305),
            #                             'datebounds':datebounds,
            #                             'season0':False,#  climo would use climoyears as bounds for years
            #                             'climoyears':climoyears,
            #                             'time_window':time_window,
            #                             'coarsegrain':resolution,
            #                             'landmask':True})},

            'T2m':
                {'info':(f'/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/offline_climatology/{resolution}p0.{climoyears[0]}-{climoyears[1]}/T2m','anomaly',
                                        {'latbounds':(20,74),
                                        'lonbounds':(190,305),
                                        'datebounds':datebounds,
                                        'climoyears':climoyears,
                                        # 'time_window':time_window, the input data has been 7-day running meaned
                                        'coarsegrain':resolution,
                                        'season0':False,
                                        'landmask':True,
                                        'get_climo':True})},
                # adding get_climo flag when making offline_climo because you need to use 7-day-running-meaned data to calculate climo
            
                }    

