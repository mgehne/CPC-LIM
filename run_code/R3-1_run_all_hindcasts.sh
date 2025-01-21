#!/bin/bash

# Loop from 10 down to 1
# for fold in {10..1}
for fold in {10..8}
do
    # Run the Python script with the fold number
    python run_for_v2p0_6_vars_hindcast_fold_${fold}.py
    # python run_for_fixed_58-16_climo_hindcast_fold_${fold}.py
done
