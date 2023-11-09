#!/bin/sh

# make sure all links are removed
#expt_name='make_rawdata_10b4_using_my_rawdata'
# expt_name='/data/ycheng/JRA/Data/make_rawdata_offline_climatology/2p0.1981-2010'
expt_name='/data/ycheng/JRA/Data/make_rawdata_offline_climatology/2p0.1971-2000'
#expt_name='make_rawdata_offline_climatology/2p0.1991-2020'

mkdir $expt_name
cd $expt_name
varnames=("hgt" "phy2m" "land" "surf" "sst" "sf")
#varnames=("sst")
for varname in "${varnames[@]}"; do
rm -rf $varname
mkdir -p $varname
cd $varname

	#for ((i = 1997; i <= 2021; i++)); do
	#for ((i = 1958; i <= 2020; i++)); do
	#for ((i = 1958; i <= 2016; i++)); do
	#for ((i = 1979; i <= 2017; i++)); do
	for ((i = 1971; i <= 2000; i++)); do 
	# for ((i = 1981; i <= 2010; i++)); do 
	#for ((i = 1991; i <= 2020; i++)); do 
    	echo $i
    	# make sure all links are removed
	if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
	    	ln -s "/data/ycheng/JRA/Data/$i/${varname}_${i}_1p25.nc" ./
	else
    		ln -s "/data/ycheng/JRA/Data/$i/${varname}_$i.nc" ./
	fi
	done
	cd ../
done
