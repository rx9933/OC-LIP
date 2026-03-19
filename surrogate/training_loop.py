import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--rM', type=int, required=True, help='Input basis dimension')
    parser.add_argument('--dQ', type=int, required=True, help='full output basis dimension')
    parser.add_argument('--dM', type=int, required=True, help='full input basis dimension')
    parser.add_argument('--n_train', type=int, required=True, help='Number of training samples')
    parser.add_argument('--n_test', type=int, required=True, help='Number of testing samples')
    parser.add_argument('--n_data', type=int, required=True, help='Number of total samples')

    args = parser.parse_args()
    large_epochs = 2000
    small_epochs = 300

    # l2
    os.system(
    f"python train_rbno_oc.py "
    f"--data_type 'xv' " # data = using the position of the drone + POD reduced velocity field
    f"--rQ {args.rQ} "
    f"--dM {args.dM} "
    f"--n_train {args.n_train} "
    f"--n_test {args.n_test} "
    f"--n_data {args.n_data} "
    f"--data_dir {args.data_dir} "
    f"--epochs {large_epochs} " 
    )

    os.system(
    f"python train_rbno_oc.py "
    f"--data_type 'xvspectral' " # data = using the position of the drone + POD reduced velocity field
    f"--rQ {args.rQ} "
    f"--dM {args.dM} "
    f"--n_train {args.n_train} "
    f"--n_test {args.n_test} "
    f"--n_data {args.n_data} "
    f"--data_dir {args.data_dir} "
    f"--epochs {large_epochs} " 
    )
    

    
    os.system(
    f"python train_fno_oc.py "
    f"--n_train {args.n_train} "
    f"--n_test {args.n_test} "
    f"--n_data {args.n_data} "
    f"--data_dir {args.data_dir} "
    f"--epochs {large_epochs} " 
    )
    
if __name__ == "__main__":
    main()