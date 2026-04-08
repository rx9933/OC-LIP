import numpy as np
import os
import subprocess
import training_loop as tl

max_n_train = 1600
n_test = 100
max_n_data = 945
trains = [800]# * (2**i) for i in range(int(np.log2(max_n_train/50)) + 1)]
for n_train in trains:
    if n_train>0:
        
        cmd = [
            'python',
            '-m', 'training_loop', 
            f'--n_train={n_train}',
            f'--n_test={n_test}',
            f'--n_data={max_n_data}',# 13000
            '--rQ=202', # currently unused, reduced data dimension
            '--dQ=202', # reduced velocity + position dimension
            '--dM=14', # parameter dimension: 4K+2 , path coefficients
            '--rM=14' # currently unused
        ]
        
        print("Executing:", ' '.join(cmd))
    

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for n_train={n_train}: {e}")
    except Exception as e:
        print(f"Unexpected error for n_train={n_train}: {e}")
