#!/bin/bash

# Get the current year
# current_year=2024
# current_year=2023
current_year=2016
# jra_data_folder="/data/ycheng/JRA/Data/"  # Data have been moved on Jan 11, 2024
jra_data_folder="/Projects/jalbers_process/CPC_LIM/yuan_ming/JRA"  
varnames=("sf" "hgt" "phy2m" "land" "surf" "sst")
# expt_name="9b2_sliding_climo_no_double_running_mean"
# expt_name="v2p0"
# expt_name="v2p0_new_make_rawdata"
# expt_name="fixed_58-16_climo"
expt_name="fixed_58-16_climo_new_make_rawdata"

# sliding_climo="True"
sliding_climo="False"

# Function to create symbolic links for a given year
create_symlinks() {
    # lim_data_dir="/data/ycheng/JRA/Data/make_rawdata_"
    lim_data_dir="/data/ycheng/JRA/Data/make_rawdata_${expt_name}"
    year=$1
    if [ "$sliding_climo" = "True" ]; then
    # This part of the script figures out the years needed for calculating sliding climo for each year and 
    # link them to the folder of the year under $varname folder.
        for ((i=1958; i<=year; i++)); do
        # for ((i=2023; i<=year; i++)); do
            echo "---------------- making $i now ----------------"
            if [ -d "$lim_data_dir/$i" ]; then
                echo "Directory '$i' already exists. Skipping."
            else
                mkdir "$lim_data_dir/$i"
                echo "Created directory '$i'."
            fi

            if [ -d "$jra_data_folder" ]; then
                # For years before 1978, create symlinks to files from 1958 to 1977
                echo $jra_data_folder
                if [ "$i" -lt 1978 ]; then
                    echo $i
                    folders_to_link=$(ls "$jra_data_folder" | grep -E "^195[8-9]|196[0-9]|197[0-7]$")
                    # a = "$grep -E "^195[8-9]|19[6-7][0-9]"
                    echo $folders_to_link
                else
                    # For years after 1978, create symlinks to files from 20 years before the current year up to the current year
                    echo $i, 'after 1978'
                    link_start_year=$(($i - 20))
                    link_end_year=$(($i)) # You need the current year because this year ${i} is what you want with climo of (${i-20}-${i-1}) 
                    # folders_to_link=$(ls "$jra_data_folder" | grep -E "^$(seq -s'|' -f"%g" "$link_start_year" "$link_end_year" | paste -sd '|')$") # for containing the string
                    folders_to_link=$(ls "$jra_data_folder" | grep -E "^($(seq -s'|' -f"%g" "$link_start_year" "$link_end_year" | paste -sd '|'))$") # for exact match
                    echo $folders_to_link

                fi
                
                # Create symlinks in the $i directory for each file
                for varname in "${varnames[@]}"; do
                    echo "---------------- $varname ----------------"
                    rm -rf $lim_data_dir/$i/$varname
                    mkdir -p $lim_data_dir/$i/$varname
                    cd $lim_data_dir/$i/$varname
                    for year_folder in $folders_to_link; do #year_folders are the year to link
                        if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
                            if [ -e "${jra_data_folder}/${year_folder}/${varname}_${year_folder}_1p25.nc" ]; then
                            # There used to be 2p5 version that needs to be distinguished. 

                                ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}_1p25.nc" "./${varname}_${year_folder}.nc"
                            else 
                            # New 2023 download is only 1p25 so not adding _1p25 suffix anymore. 
                            # Need to change the raw file name to not have _1p25 
                                echo "linking ${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc"
                                ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc" ./
                            fi

                        else
                            ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc" ./
                        fi
                        echo ${year_folder}
                    done
                    # Kept this option to link real time data
                    if [ "$i" -eq 2024 ]; then # use the 2024 files with realtime data
                        ln -sf "${jra_data_folder}/${i}_realtime/${varname}_${i}.nc" ./
                    fi                 
                    
                    cd $lim_data_dir/$i/
                done
            else
                echo "Error: Directory '$jra_data_folder' not found. Skipping linking for '$i'."
            fi
        done

    elif [ "$sliding_climo" = "False" ]; then
    # For fixed climo, we just link all the years up to "$current_year" in one folder named $varname
        echo "Not using sliding climo, just link all years in one folder"
        if [ -d "$lim_data_dir/" ]; then
            echo "Directory '$i' already exists. Skipping."
        else
            mkdir "$lim_data_dir/"
            echo "Created directory '$lim_data_dir'."
        fi
        folders_to_link=$(ls "$jra_data_folder")
        # a = "$grep -E "^195[8-9]|19[6-7][0-9]"
        echo $folders_to_link
        for folder in $folders_to_link; do
            # Extract the first four digits from the folder name
            year_folder=${folder:0:4}

            # Check if the year is less than 2022
            if [ "$year_folder" -le "$current_year" ]; then
                # year_folder = "${jra_data_folder}/${year}"
                echo "$year_folder is from before 2022"
                for varname in "${varnames[@]}"; do
                    echo "---------------- $varname ----------------"
                    # echo $lim_data_dir/$varname
                    if [ "$year_folder" = "1958" ]; then
                        rm -rf $lim_data_dir/$varname
                    fi
                    mkdir -p $lim_data_dir/$varname
                    cd $lim_data_dir/$varname
                    if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
                        if [ -e "${jra_data_folder}/${year_folder}/${varname}_${year_folder}_1p25.nc" ]; then
                        # There used to be 2p5 version that needs to be distinguished. 

                            ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}_1p25.nc" "./${varname}_${year_folder}.nc"
                        else 
                        # New 2023 download is only 1p25 so not adding _1p25 suffix anymore. 
                        # Need to change the raw file name to not have _1p25 
                            echo "linking ${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc"
                            ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc" ./
                        fi

                    else
                        ln -s "${jra_data_folder}/${year_folder}/${varname}_${year_folder}.nc" ./
                    fi
                done
            else
                echo "$year_folder is later than $current_year"
            fi
        done
        
    fi
}

# Create directories and symlinks up to the current year
create_symlinks "$current_year"
