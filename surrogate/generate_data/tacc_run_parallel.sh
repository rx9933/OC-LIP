File Edit Options Buffers Tools Sh-Script Help
#!/bin/bash

export OMP_NUM_THREADS=1

log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_time "Starting job ${SLURM_ARRAY_TASK_ID}"

total_samples=4
num_jobs=2

samples_per_job=$((total_samples / num_jobs))
remainder=$((total_samples % num_jobs))

job_id=${SLURM_ARRAY_TASK_ID}

# Handle uneven split
extra=0
if [ $job_id -lt $remainder ]; then
    extra=1
fi

current_samples=$((samples_per_job + extra))

log_time "Job ${job_id} processing ${current_samples} samples"

mkdir -p logs

python training_data_generator_hippylib_example.py \
    --job_id ${job_id} \
    --n_samples ${current_samples} \
    --output_prefix "oed_training_data_${job_id}" \
    > logs/log_job${job_id}.txt 2>&1

log_time "Job ${job_id} finished"



