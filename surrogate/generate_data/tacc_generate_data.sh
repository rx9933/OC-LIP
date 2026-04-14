File Edit Options Buffers Tools Sh-Script Help
#!/bin/bash

#SBATCH -J generate_data
#SBATCH -o logs/out_%A_%a.out
#SBATCH -e logs/err_%A_%a.err
#SBATCH -p gg # maybe can do 20 on gh simultaneously?
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -A FTA-SUB-Ghattas
#SBATCH --array=0-19
#SBATCH --exclusive

source ~/.bashrc
conda activate torchfem

bash tacc_run_parallel.sh



