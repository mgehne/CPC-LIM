#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:43:52 2021

@author: slillo
"""

from datetime import datetime as dt,timedelta
import xarray as xr
import requests

year = dt.now().year

for yr in (year-1,year):
    for tlab in ('tmax','tmin'):
        try:
            url = f'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/{tlab}.{yr}.nc'
            r = requests.get(url)
            open(f'data_realtime/cpc{tlab}{yr}.nc', 'wb').write(r.content)
        except:
            print(f'couldnt get {tlab} {yr}')
    

ds = xr.open_mfdataset([f'data_realtime/cpctmin{yr}.nc',f'data_realtime/cpctmax{yr}.nc'])