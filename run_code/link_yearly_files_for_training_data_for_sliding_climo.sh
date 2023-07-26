#!/bin/bash
in_data_folder="/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo"
out_data_folder="/Projects/jalbers_process/CPC_LIM/yuan_ming/Data/9_sliding_climo"

varnames=("T2m" "SOIL" "SLP" "colIrr" "H500" "SST" "SF100" "SF750")
end_year=2016
# end_year=1959
for ((i=1958; i<=end_year; i++)); do
    for varname in "${varnames[@]}"; do
        echo "---------------- making $i for $varname now ----------------"
        mkdir -p "$out_data_folder/$varname"
        if [ -f "$in_data_folder/$i/$varname/${varname}.${i}.nc" ]; then
            ln -sf "$in_data_folder/$i/$varname/${varname}.${i}.nc" "$out_data_folder/$varname"
        else 
            echo "!!!missing $in_data_folder/$i/$varname/${varname}.${i}.nc !!!!!"
        fi
    done

done

