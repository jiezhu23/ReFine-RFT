TASKS=("test" "val")
# Adjust to your gpu num
GPU_IDS=(0 1 2 3 4 5 6 7)
SPLIT_NUM=8

for task in "${TASKS[@]}"; do
    echo "Starting inference for task: $task"

    # 遍历 GPU 和 SPLIT
    for i in "${!GPU_IDS[@]}"; do
        GPU_ID=${GPU_IDS[$i]}
        SPLIT=$i
        echo "Launching task=$task on GPU=$GPU_ID with SPLIT=$SPLIT"
        SPLIT=$SPLIT SPLIT_NUM=$SPLIT_NUM python Qwen2_VL_lisa_infere.py \
            --task $task &
        sleep 1
    done
    wait
    echo "Merging results for task: $task"
    SPLIT_NUM=$SPLIT_NUM python merge_eval.py >> res.txt
done

echo "All tasks completed!"
