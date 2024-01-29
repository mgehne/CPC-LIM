#!/bin/sh
# This script links the JRA raw files needed to calculate climatology. It will create symbolic links for 30-year of data for each variables.
jra_data_folder="/Projects/jalbers_process/CPC_LIM/yuan_ming/JRA"

# expt_name="2p0.1971-2000"
# expt_name="2p0.1981-2010"
expt_name="2p0.1991-2020"

make_climatology_dir='/data/ycheng/JRA/Data/make_rawdata_climatology'
make_data_dir="${make_climatology_dir}/${expt_name}"

mkdir -p $make_climatology_dir
mkdir $make_data_dir
cd $make_data_dir
varnames=("hgt" "phy2m" "land" "surf" "sst" "sf")
#varnames=("sst")
for varname in "${varnames[@]}"; do
rm -rf $varname
mkdir -p $varname
cd $varname

# for ((i = 1971; i <= 2000; i++)); do 
# for ((i = 1981; i <= 2010; i++)); do 
for ((i = 1991; i <= 2020; i++)); do 
	echo $i
	# make sure all links are removed
if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
		ln -s ln -s "${jra_data_folder}/${i}/${varname}_${i}_1p25.nc" ./
else
		ln -s "${jra_data_folder}/${i}/${varname}_${i}.nc" ./
fi
done
cd ../

done
