#!/bin/bash
video_path=${1:-cases/sample1_.webm}
cuda=${2:-0}

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
    --dirname $output_dir \
    --mode eqfov \

#end
