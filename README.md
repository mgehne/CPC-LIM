# CPC-LIM
Create realtime forecasts for the NOAA PSL/CPC subseasonal LIM.

Getting started:

1) Run the bash shell script to to set up the environment.
2) Get data_clim, rawdata and data_realtime directories from FTP (ftp://ftp2.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/) and put those in the directory run_code/. User should end up with directory structure run_code/data_clim, run_code/rawdata etc. (Alternatively data can be downloaded via: https://downloads.psl.noaa.gov/Projects/LIM/Realtime/Realtime/webData/)
3) At the same level as the run_code directory make a directories called Images and lim_s2s.
4) Set the directory to copy the .png and .html files to on line 80 in run_for_realtime.py. The default is to put final images in the directory lim_s2s created above.
5) Run the realtime forecast with: python3 run_realtime.py
