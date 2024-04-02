#!/bin/sh
# This script links the JRA raw files needed to calculate climatology. It will create symbolic links for 30-year of data for each variables.
# source_data_folder="/Projects/jalbers_process/CPC_LIM/yuan_ming/JRA"
# CPCdata=false

source_data_folder='/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/climatology_cpc/data'
CPCdata=true

# expt_name="2p0.1971-2000"
# expt_name="2p0.1981-2010"
expt_name="2p0.1991-2020"

make_climatology_dir='/data/ycheng/JRA/Data/make_rawdata_climatology_CPC'
# make_climatology_dir='/Users/ycheng/CPC/Data/make_rawdata_climatology_CPC'
make_data_dir="${make_climatology_dir}/${expt_name}"

mkdir -p $make_climatology_dir
mkdir -p $make_data_dir
cd $make_data_dir

if [ "$CPCdata" = true ]; then
	varnames=("tavg")
else 
	varnames=("hgt" "phy2m" "land" "surf" "sst" "sf")
fi
#varnames=("sst")
for varname in "${varnames[@]}"; do
	# make sure all links are removed
	rm -rf $varname
	mkdir -p $varname
	cd $varname

	# for ((i = 1971; i <= 2000; i++)); do 
	# for ((i = 1981; i <= 2010; i++)); do 
	for ((i = 1991; i <= 2020; i++)); do 
		echo $i
		if [ "$CPCdata" = true ]; then
			ln -s "${source_data_folder}/${varname}.${i}.2p0.nc" ./

		else
			if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
					ln -s ln -s "${source_data_folder}/${i}/${varname}_${i}_1p25.nc" ./
			else
					ln -s "${source_data_folder}/${i}/${varname}_${i}.nc" ./
			fi
		fi

	done
	cd ../

done
