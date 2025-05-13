#!/bin/bash

input_csv="" # input video ids
output_dir="" # output directory
jobs=8 # number of jobs
log_name="download.log" # log file name

python download_list.py \
    -i "${input_csv}" \
    -o "${output_dir}" \
    -j ${jobs} \
    > ${log_name}
