#!/bin/bash
video_path=$1
checkpoint_dir=$2 # checkpoint dir
cuda=$3 # cuda id

model_config="configs/v2a_foa.json"

base_name="demo"

result_base="results"

output_dir="${result_base}/${base_name}"
mkdir -p $output_dir

set -x

CUDA_VISIBLE_DEVICES=$cuda \
    python inference.py \
    --base-dir $video_path \
    --infer-type v2a \
    --model-config $model_config \
    --ckpt-path $checkpoint_dir \
    --dirname $output_dir \
    --mode eqfov \

#end
