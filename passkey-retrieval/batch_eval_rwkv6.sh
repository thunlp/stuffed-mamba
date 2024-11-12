output_dir="./output/${model_name}/${run_name}"
min_len="1"
max_len="16"

cmd="python eval_rwkv6.py"
cmd+=" --min_len ${min_len} --max_len ${max_len}"
cmd+=" --n_gpus 8 --multi_gpu 1"

model_path="/path/to/model"
output_path="./output/rwkv6"

cmd+=" --model_path ${model_path} --tok_path ${model_path} --output_dir ${output_path}"

echo "================="
echo "$cmd"
echo "================="

gpu_ids=(0 1 2 3 4 5 6 7)
pids=()
# Loop over the GPU IDs
for gpu_id in "${gpu_ids[@]}"; do
    # Execute the command with the current GPU ID
    this_gpu_cmd="${cmd} --gpu_id ${gpu_id} --device cuda:${gpu_id}"
    echo "$this_gpu_cmd"
    $this_gpu_cmd &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "DONE"
