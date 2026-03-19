#!/bin/bash

export OMP_NUM_THREADS=1

# Function to print timestamped messages
log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

PROJECT_NAME='pml' # or free_surface
log_time "Starting parallel data generation..."

# Total samples needed
total_samples=80 #12800

num_jobs=40 
samples_per_job=$((total_samples / num_jobs))

log_time "Total samples: $total_samples"
log_time "Samples per job: $samples_per_job"
log_time "Number of jobs: $num_jobs"

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch jobs
for ((job_id=0; job_id<num_jobs; job_id++)); do
    log_time "Launching job ${job_id} with ${samples_per_job} samples"
    
    python training_data_generator.py \
        --job_id ${job_id} \
        --n_samples ${samples_per_job} \
        --output_prefix "oed_training_data" > logs/log_job${job_id}.txt 2>&1 &
    
    pid=$!
    log_time "Job ${job_id} started with PID ${pid}"
    
    # Optional: small delay between job launches to avoid overwhelming the system
    sleep 1
done

log_time "Waiting for all jobs to complete..."
wait
log_time "All jobs finished"

# Combine all the output files
log_time "Combining output files..."
python -c "
import pickle
import glob

all_data = []
for f in sorted(glob.glob('oed_training_data_job*.pkl')):
    print(f'Loading {f}')
    with open(f, 'rb') as fp:
        data = pickle.load(fp)
        all_data.extend(data)

print(f'Total samples: {len(all_data)}')
with open('oed_training_data_combined.pkl', 'wb') as f:
    pickle.dump(all_data, f)
print('Saved combined data to oed_training_data_combined.pkl')

for f in sorted(glob.glob('oed_training_data_job*.pkl')):
    os.remove(f)
    print(f'Deleted {f}')

"

log_time "Done!"