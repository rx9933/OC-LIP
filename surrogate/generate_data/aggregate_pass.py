"""
aggregate_pass.py
Combines per-sample pickles into one labels .npz. Run after each pass.
"""

import os, glob, pickle, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pass_num', type=int, required=True)
parser.add_argument('--work_dir', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=7957)
args = parser.parse_args()

print("=" * 70)
print(f"  AGGREGATING PASS {args.pass_num}")
print("=" * 70)

files = sorted(glob.glob(os.path.join(args.work_dir, 'sample_*.pkl')))
print(f"  Found {len(files)} per-sample files in {args.work_dir}")
print(f"  Expected {args.n_samples}")
if len(files) < args.n_samples:
    print(f"  WARNING: {args.n_samples - len(files)} samples missing!")

prev_file = args.output_file.replace(f'pass{args.pass_num}', f'pass{args.pass_num-1}')
prev = np.load(prev_file)
m_new = prev['m'].copy()
eig_new = prev['eig'].copy()

n_improved = 0
improvements = []
sources = {'self': 0, 'original': 0}
for k in range(10):
    sources[f'neighbor_{k}'] = 0

for f in files:
    with open(f, 'rb') as h:
        r = pickle.load(h)
    i = r['sample_idx']
    m_new[i] = r['m_new']
    eig_new[i] = r['eig_new']
    if r['improved']:
        n_improved += 1
        improvements.append(r['improvement'])
    sources[r['best_source']] = sources.get(r['best_source'], 0) + 1

np.savez(args.output_file, m=m_new, eig=eig_new)
print(f"\n  Saved: {args.output_file}")

print(f"\n  Pass {args.pass_num} stats:")
print(f"    Improved samples: {n_improved} ({100*n_improved/len(files):.1f}%)")
if improvements:
    imp = np.array(improvements)
    print(f"    Mean improvement (among improved): +{imp.mean():.3f} EIG")
    print(f"    Max improvement: +{imp.max():.3f} EIG")
    print(f"    Total EIG gain across dataset: +{imp.sum():.1f}")

print(f"\n  Improvement sources:")
for src, cnt in sorted(sources.items(), key=lambda kv: -kv[1]):
    if cnt > 0:
        print(f"    {src:<15s}: {cnt:5d} ({100*cnt/len(files):.1f}%)")

print(f"\n  Dataset EIG: mean={eig_new.mean():.3f} (was {prev['eig'].mean():.3f})")
print("=" * 70)
