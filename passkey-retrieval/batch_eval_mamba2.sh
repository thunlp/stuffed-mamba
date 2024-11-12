working_dir=".."
run_name="run-name"
model_path="/path/to/model"
output_path="./output/mamba2-370m"
model_name="mamba2-370m"
ckpt_dir="${working_dir}/ckpts/${model_name}/${run_name}"
output_dir="./output/${model_name}/${run_name}"
tok_path="${working_dir}/tokenizers/mamba-tok"

cmd="python eval_mamba2.py "
cmd+=" --model_name mamba2"
cmd+=" --tok_path ${tok_path}"
cmd+=" --min_len 1"
cmd+=" --max_len 128"
cmd+=" --n_gpus 8"
cmd+=" --model_path ${model_path}"
cmd+=" --output_dir ${output_path}"

# Loop over the GPU IDs
for gpu_id in "${gpu_ids[@]}"; do
    # Execute the command with the current GPU ID
    this_gpu_cmd="${cmd} --gpu_id ${gpu_id} &"

    echo "$this_gpu_cmd"

    eval "$this_gpu_cmd"
done

wait
