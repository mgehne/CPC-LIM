#!/bin/bash

# Define the range of years
start_year=2014
end_year=2014
for ((year=start_year; year<=end_year; year++))
do 
    # wget -O "data_retrospective/cpctmax.${year}.nc" "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.${year}.nc"
    # wget -O "data_retrospective/cpctmin.${year}.nc" "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.${year}.nc"
    wget -O "data_realtime/cpctmax.${year}.nc" "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.${year}.nc"
    wget -O "data_realtime/cpctmin.${year}.nc" "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.${year}.nc"
done