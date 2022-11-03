#!/bin/sh
# This script builds a Python environment for running the NOAA/PSL Experimental LIM Forecast model (v1.2)

echo "Building Python environment..."

echo""

conda config --add channels conda-forge
conda create -y --name cpc_lim_v1.2_env
conda activate cpc_lim_v1.2_env
conda install -y matplotlib=3.5.2
conda install -y scipy
conda install -y basemap
conda install -y netCDF4
conda install -y xarray
conda install -y regionmask
conda install -y cartopy
conda install -y cfgrib
conda install -y dask
pip install global_land_mask

echo""
