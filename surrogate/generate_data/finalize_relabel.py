"""
finalize_relabel.py
Merges relabeled m/eig back into a full dataset file compatible with train_nn.py.
Output: data/mq_data_relabeled.npz
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--original_file', type=str, default='data/mq_data_reduced.npz')
parser.add_argument('--final_labels', type=str, required=True)
parser.add_argument('--output_file', type=str, default='data/mq_data_relabeled.npz')
args = parser.parse_args()

print("=" * 70)
print("  FINALIZE RELABELING")
print("=" * 70)

orig = np.load(args.original_file, allow_pickle=True)
new = np.load(args.final_labels)

out = {k: orig[k] for k in orig.files}
out['m'] = new['m']
out['eig_K3'] = new['eig']

print(f"  Original EIG: mean={orig['eig_K3'].mean():.3f}, "
      f"min={orig['eig_K3'].min():.3f}, max={orig['eig_K3'].max():.3f}")
print(f"  Relabel EIG:  mean={new['eig'].mean():.3f}, "
      f"min={new['eig'].min():.3f}, max={new['eig'].max():.3f}")
print(f"  Net gain: +{new['eig'].mean() - orig['eig_K3'].mean():.3f} avg EIG")

np.savez(args.output_file, **out)
print(f"\n  Saved: {args.output_file}")
print("=" * 70)
