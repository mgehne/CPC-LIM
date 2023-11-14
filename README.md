# CPC-LIM
Create realtime forecasts for the NOAA PSL/CPC subseasonal LIM.

Getting started:

1) Run the shell script cpc_lim_v2.0_env_build.sh to to set up the environment. In order to build correctly, you need to type ‘source cpc_lim_v2.0_env_build.sh’ from the command line so that conda is linked correctly. The user needs to have conda installed on their system in order to do this. Building the environment can take a while.
Note: Users need to make sure that the environment gets built without errors. If there are error messages during the build, even if it finishes, the code will likely not run correctly. 
2) Run the shell script cpc_lim_data_download.sh to get the necessary data directories. The script should download the data_clim directories from FTP (ftp://ftp2.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/v2.0/) and then place them in the correct location. If this works correctly, then the data_clim directories should be located inside of the directory /run_code (e.g.,  directory structure run_code/data_clim). (Alternatively data can be downloaded via: https://downloads.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/).
3) Set the directory to copy the .png and .html files to on line 43 in run_for_realtime.py. 
4) Activate the conda environment.
5) Run the realtime forecast with: python3 run_for_realtime.py (you must be in the /run_code directory to run this)
6) To rerun dates that already have output in Images/yyyymmdd delete the yyyymmdd directory before running again. Otherwise the output will not be overwritten.
