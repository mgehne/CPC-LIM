#!/bin/bash

# Get the current year
current_year=2022
data_folder="/data/ycheng/JRA/Data/"  
varnames=("sf" "hgt" "phy2m" "land" "surf" "sst")
# Function to create symbolic links for a given year
create_symlinks() {
    home_dir="/data/ycheng/JRA/Data/make_rawdata_9_sliding_climo"
    year=$1
    for ((i=1958; i<=year; i++)); do
        echo "---------------- making $i now ----------------"
        if [ -d "$home_dir/$i" ]; then
            echo "Directory '$i' already exists. Skipping."
        else
            mkdir "$home_dir/$i"
            echo "Created directory '$i'."
        fi

        if [ -d "$data_folder" ]; then
            # For years before 1978, create symlinks to files from 1958 to 1977
            echo $data_folder
            if [ "$i" -lt 1978 ]; then
                echo $i
                folders_to_link=$(ls "$data_folder" | grep -E "^195[8-9]|196[0-9]|197[0-7]$")
                # a = "$grep -E "^195[8-9]|19[6-7][0-9]"
                echo $folders_to_link
            else
                # For years after 1978, create symlinks to files from 20 years before the current year up to the current year
                echo $i, 'after 1978'
                link_start_year=$(($i - 20))
                link_end_year=$(($i)) # You need the current year because this year ${i} is what you want with climo of (${i-20}-${i-1}) 
                folders_to_link=$(ls "$data_folder" | grep -E "^$(seq -s'|' -f"%g" "$link_start_year" "$link_end_year" | paste -sd '|')$")
                echo $folders_to_link

            fi
            
            # Create symlinks in the $i directory for each file
            for varname in "${varnames[@]}"; do
                echo "---------------- $varname ----------------"
                rm -rf $home_dir/$i/$varname
                mkdir -p $home_dir/$i/$varname
                cd $home_dir/$i/$varname
                for year_folder in $folders_to_link; do #year_folders are the year to link
                    if [ "$varname" = "sf" ] || [ "$varname" = "hgt" ]; then
                        ln -s "/data/ycheng/JRA/Data/$year_folder/${varname}_${year_folder}_1p25.nc" ./
                    else
                        ln -s "/data/ycheng/JRA/Data/$year_folder/${varname}_${year_folder}.nc" ./
                    fi
                    echo ${year_folder}
                done
                cd $home_dir/$i/
            done
        else
            echo "Error: Directory '$data_folder' not found. Skipping linking for '$i'."
        fi
    done
}

# Create directories and symlinks up to the current year
create_symlinks "$current_year"
