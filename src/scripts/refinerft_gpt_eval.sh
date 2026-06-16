PYTHON_PATH=~/anaconda3/envs/refinerft/bin/python


${PYTHON_PATH} src/refinerft/data/openai_fgvc_eval.py \
    --dataset_name laolao77/ViRFT_CLS_fgvc_aircraft_4_shot \
    --output_dir ./src/refinerft/data/ \
    --max_sample 150 \
    --num_slice 5 \
    --prompt_type cot \