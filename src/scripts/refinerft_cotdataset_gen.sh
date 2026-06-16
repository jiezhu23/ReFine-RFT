PYTHON_PATH=~/anaconda3/envs/refinerft/bin/python


${PYTHON_PATH} src/refinerft/data/create_cotdataset_hf.py \
    --dataset_name laolao77/ViRFT_CLS_flower_4_shot \
    --output_dir ./src/refinerft/data/ \
    --max_sample 150 \
    --num_slice 5 \