#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:13:51 2021

@author: slillo

Edited: J.R. Albers 10.4.2022
This function is used to create retrospective (out-of-sample) reforecasts using the NOAA PSL/CPC subseasonal LIM.

- Forecasts are saved as netCDF files via the directories LIMpage_path and FCSTDIR
- LIM forecast operator pickles must already have been created; if they haven't, then read=False must be inserted into the LIMdriver.get_variables() and LIMdriver.get_eofs() calls
- As currently set up, the forecasts use the add_offset flag, which adjusts the reference climatology of the anomalies to be that of the current NOAA CPC base period (currently 1991-2020)

"""

####################################################################################
# IMPORT PACKAGES
####################################################################################
import numpy as np
from datetime import datetime as dt,timedelta
import xarray as xr
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from calendar import monthrange

# Edited import method J.R. ALbers 10.4.2022
import lib
from lib import driver
from lib import data_retrieval
from lib import dataset
from lib import model
from lib import plot
from lib import tools
from lib import verif
from lib.tools import *
# from LIM_CPC import driver
# import data_retrieval
# import LIM_CPC
# from LIM_CPC.tools import *

import warnings
warnings.filterwarnings('ignore')


####################################################################################
### BEGIN USER INPUT ###

#LIMpage_path = f'../Images'
LIMpage_path = f'../Images_retrospective'
FCSTDIR = f'{LIMpage_path}/lim_t2m_retrospective/wk34separate_beta'

### END USER INPUT ###
####################################################################################


####################################################################################
# START RUN CODE
####################################################################################

#for YEAR in range(2017,2022):
#    for MONTH in range(1,13):

# INITIALIZE AND RUN LIM FORECAST
print('\nInitializing and running LIM...')
#LIMdriver = driver.Driver(f'namelist_{YEAR}{MONTH:02}.py')
LIMdriver = driver.Driver(f'namelist.py')
LIMdriver.get_variables()
LIMdriver.get_eofs()
LIMdriver.prep_realtime_data(limkey=1)

#LASTDAY = max(monthrange(YEAR,MONTH))
T_START = dt(2022,10,1) #dt(YEAR,MONTH,1)
T_END = dt(2022,10,29) #dt(YEAR,MONTH,LASTDAY)
hindcastdays = [T_START + timedelta(days=i) for i in range((T_END-T_START).days+1)]

for T_INIT in hindcastdays:
    START = dt.now()
    try:
        LIMdriver.run_forecast_blend(t_init=T_INIT,lead_times=(21,28),fullVariance=True)
        LIMdriver.save_netcdf_files(varname='T2m',t_init=T_INIT,lead_times=(21,28),save_to_path=FCSTDIR,add_offset='data_clim/CPC.1991-2020.nc')
    except:
        print(f'{T_INIT:%Y%m%d} data is unavailable and/or forecast was unable to run')
        pass

    FINISH = dt.now()
    ELAPSED = (FINISH-START).total_seconds()/60
    print(f'\n {ELAPSED:.2f} minutes to run {T_INIT:%Y%m%d}')
