"""
prepare_relabel.py
Computes nearest neighbors for all samples. Run ONCE before pass 1.
Saves:
  data/relabel_neighbors.npz       (nn_idxs: (N, n_neighbors))
  data/relabel_labels_pass0.npz    (m, eig: starting point for pass 1)
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/mq_data_reduced.npz')
parser.add_argument('--output_dir', type=str, default='data/')
parser.add_argument('--n_neighbors', type=int, default=10)
parser.add_argument('--n_pod', type=int, default=2)
args = parser.parse_args()

print("=" * 70)
print("  PREPARE RELABELING PIPELINE")
print("=" * 70)

d = np.load(args.input_file, allow_pickle=True)
x = d['x']
v = d['v'][:, :args.n_pod]
m = d['m']
eig_K3 = d['eig_K3']
N = len(x)
print(f"  Samples: {N}")

v_std = v.std(axis=0)
v_std[v_std < 1e-8] = 1.0
v_normed = v / v_std

inputs = np.concatenate([x, v_normed], axis=1)
print(f"  Input space: {inputs.shape[1]}D  (c0 + {args.n_pod} POD normalized)")

print(f"  Computing {args.n_neighbors} nearest neighbors per sample...")
tree = cKDTree(inputs)
dists, idxs = tree.query(inputs, k=args.n_neighbors + 1)
nn_idxs = idxs[:, 1:]
nn_dists = dists[:, 1:]

print(f"  Mean distance to nearest neighbor: {nn_dists[:, 0].mean():.4f}")
print(f"  Mean distance to {args.n_neighbors}th neighbor: {nn_dists[:, -1].mean():.4f}")

nbr_path = f'{args.output_dir}/relabel_neighbors.npz'
np.savez(nbr_path, nn_idxs=nn_idxs, nn_dists=nn_dists)
print(f"  Saved: {nbr_path}")

lbl_path = f'{args.output_dir}/relabel_labels_pass0.npz'
np.savez(lbl_path, m=m.copy(), eig=eig_K3.copy())
print(f"  Saved: {lbl_path}  (initial labels = original)")

print("=" * 70)
print("  READY TO START PASS 1")
print("=" * 70)
