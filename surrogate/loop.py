import numpy as np
import os
import subprocess
import training_loop as tl

max_n_train = 12800
n_test = 200
trains = [50 * (2**i) for i in range(int(np.log2(max_n_train/50)) + 1)][-1:]
for n_train in trains:
    if n_train>0:
        
        cmd = [
            'python',
            '-m', 'training_loop', 
            f'--n_train={n_train}',
            f'--n_test={n_test}',
            f'--n_data={max_n_train+n_test}',# 13000
            '--rQ=100', # reduced data dimension
            '--dQ=100',
            '--dM=1661', # parameter dimension: 4K+2 , path coefficients
        ]
        
        print("Executing:", ' '.join(cmd))
    

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for n_train={n_train}: {e}")
    except Exception as e:
        print(f"Unexpected error for n_train={n_train}: {e}")
