# CPC-LIM
Create realtime forecasts for the NOAA PSL/CPC subseasonal LIM.

Getting started:

1) Run the shell script cpc_lim_v1.2_env_build.sh to to set up the environment. In order to build correctly, you need to type ‘source cpc_lim_v1.2_env_build.sh’ from the command line so that conda is linked correctly. Building the environment can take a while.
2) Run the shell script cpc_lim_data_download.sh to get the necessary data directories. The script should download the data_clim, rawdata and data_realtime directories from FTP (ftp://ftp2.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/) and then place them in the correct location. If this works correctly, then the data_clim, rawdata, and data_realtime directories should be located inside of the directory /run_code (e.g.,  directory structure run_code/data_clim, run_code/rawdata etc. (Alternatively data can be downloaded via: https://downloads.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/).
3) At the same level as the run_code directory make directories called Images and lim_s2s. The Images directory will contain the plots from the model run.
4) Set the directory to copy the .png and .html files to on line 80 in run_for_realtime.py. The default is to put final images in the directory lim_s2s created above.
5) Run the realtime forecast with: python3 run_for_realtime.py
6) To rerun dates that already have output in Images/yyyymmdd delete the yyyymmdd directory before running again. Otherwise the output will not be overwritten.
