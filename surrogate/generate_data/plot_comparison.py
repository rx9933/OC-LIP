"""
Plot comparison: multi-start vs multi-refinement results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dolfin as dl
import sys
import os

sys.path.append(os.environ.get('HIPPYLIB_BASE_DIR', "../"))
from hippylib import *

sys.path.append('../../')
from model_ad_diff_bwd import TimeDependentAD

from fourier_utils import generate_targets, fourier_frequencies, fourier_velocity
from wind_utils import sample_spectral_wind
from fe_setup import setup_fe_spaces, setup_prior, setup_true_initial_condition
from config import TY, K, OBSERVATION_TIMES, SIMULATION_TIMES, V_MAX

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

OUTDIR = 'multi_start_plots'
os.makedirs(OUTDIR, exist_ok=True)


def load_multi_start(filename='multi_start_results.pkl'):
    print(f"Loading multi-start: {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['results'])} runs")
    return data


def load_multi_refinement(filename='multi_refinement_results.pkl'):
    print(f"Loading multi-refinement: {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['results'])} runs")
    return data


def reconstruct_wind(wind_coeffs, mesh, wind_seed=None):
    from wind_utils import construct_curved_wind
    if 'pattern' in wind_coeffs:
        return construct_curved_wind(
            mesh, pattern=wind_coeffs['pattern'],
            strength=wind_coeffs['strength'],
            center=wind_coeffs['center'],
            mean_vx=wind_coeffs['mean_vx']
        )
    elif wind_coeffs.get('type') == 'opposing_inlets':
        from wind_utils import compute_velocity_field_opposing_inlets
        return compute_velocity_field_opposing_inlets(
            mesh, speed_left=wind_coeffs['speed_left'],
            speed_top=wind_coeffs['speed_top'],
            Re_val=wind_coeffs['Re']
        )
    elif wind_coeffs.get('type') == 'navier_stokes':
        from wind_utils import compute_velocity_field_navier_stokes
        return compute_velocity_field_navier_stokes(
            mesh, direction=wind_coeffs['direction'],
            speed=wind_coeffs['speed'],
            Re_val=wind_coeffs['Re']
        )
    else:
        v, _ = sample_spectral_wind(
            mesh, r_wind=wind_coeffs['r_wind'], sigma=wind_coeffs['sigma'],
            alpha=wind_coeffs['alpha'], mean_vx=wind_coeffs['mean_vx'],
            mean_vy=wind_coeffs['mean_vy'], seed=wind_seed
        )
        return v


def print_comparison_table(ms_data, mr_data):
    ms_results = ms_data['results']
    mr_results = mr_data['results']
    c0 = ms_data['c0']

    ms_eigs = [r['eig_opt'] for r in ms_results]
    mr_eigs = [r['eig_K3'] for r in mr_results]

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: MULTI-START vs MULTI-REFINEMENT")
    print(f"  c0 = ({c0[0]:.2f}, {c0[1]:.2f})")
    print(f"{'=' * 80}")

    print(f"\n  {'Run':>4s}  {'MS EIG_opt':>10s}  {'MR K=1':>8s}  {'MR K=2':>8s}  "
          f"{'MR K=3':>8s}  {'MS-MR diff':>10s}")
    print(f"  {'-' * 58}")

    n = min(len(ms_results), len(mr_results))
    for i in range(n):
        ms_eig = ms_results[i]['eig_opt']
        mr_k1 = mr_results[i]['eig_K1']
        mr_k2 = mr_results[i]['eig_K2']
        mr_k3 = mr_results[i]['eig_K3']
        diff = ms_eig - mr_k3
        print(f"  {i+1:4d}  {ms_eig:10.2f}  {mr_k1:8.2f}  {mr_k2:8.2f}  "
              f"{mr_k3:8.2f}  {diff:+10.2f}")

    print(f"\n  {'':30s}  {'Multi-start':>12s}  {'Multi-refine':>12s}")
    print(f"  {'-' * 58}")
    print(f"  {'Best EIG':30s}  {max(ms_eigs):12.2f}  {max(mr_eigs):12.2f}")
    print(f"  {'Worst EIG':30s}  {min(ms_eigs):12.2f}  {min(mr_eigs):12.2f}")
    print(f"  {'EIG spread':30s}  {max(ms_eigs)-min(ms_eigs):12.2f}  {max(mr_eigs)-min(mr_eigs):12.2f}")
    print(f"  {'EIG mean':30s}  {np.mean(ms_eigs):12.2f}  {np.mean(mr_eigs):12.2f}")
    print(f"  {'EIG std':30s}  {np.std(ms_eigs):12.2f}  {np.std(mr_eigs):12.2f}")
    print(f"{'=' * 80}")


def plot_refinement_per_run(mr_data, coords, ic_arr, c0):
    results = mr_data['results']
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)

    for i, r in enumerate(results):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                        levels=20, cmap='viridis', alpha=0.6)

        stage_colors = ['red', 'orange', 'blue']
        stage_labels = ['K=1', 'K=2', 'K=3']

        for s_idx, stage in enumerate(r['stages']):
            K_stage = stage['K']
            omegas_stage = fourier_frequencies(TY, K_stage)
            path = generate_targets(stage['m_opt'], t_dense, K_stage, omegas_stage)
            sensors = generate_targets(stage['m_opt'], OBSERVATION_TIMES, K_stage, omegas_stage)

            lw = 1.5 if s_idx < 2 else 2.5
            alpha = 0.5 if s_idx < 2 else 0.9
            ls = '--' if s_idx < 2 else '-'

            ax.plot(path[:, 0], path[:, 1], ls, color=stage_colors[s_idx],
                    lw=lw, alpha=alpha,
                    label=f'{stage_labels[s_idx]} (EIG={stage["eig_opt"]:.2f})')

            if s_idx == 2:
                n_s = len(sensors)
                ax.scatter(sensors[:, 0], sensors[:, 1], c=range(n_s),
                           cmap='coolwarm', s=30, alpha=0.8, edgecolors='black',
                           linewidths=0.5, zorder=5)

        ax.scatter(c0[0], c0[1], s=150, marker='*', c='yellow',
                   edgecolors='black', linewidths=1.5, zorder=10,
                   label=f'c0=({c0[0]},{c0[1]})')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
        gain = r['eig_K3'] - r['eig_K1']
        ax.set_title(f'Run {i+1}: K=1 -> K=2 -> K=3\n'
                     f'EIG: {r["eig_K1"]:.2f} -> {r["eig_K2"]:.2f} -> {r["eig_K3"]:.2f} '
                     f'(gain=+{gain:.2f})', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        fname = f'{OUTDIR}/refinement_run_{i+1:02d}.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fname}")


def plot_refinement_progression_grid(mr_data, coords, ic_arr, c0):
    results = mr_data['results']
    n = len(results)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)

    n_cols = min(5, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    stage_colors = ['red', 'orange', 'blue']

    for i, r in enumerate(results):
        ax = axes[i // n_cols, i % n_cols]
        ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                        levels=15, cmap='viridis', alpha=0.5)

        for s_idx, stage in enumerate(r['stages']):
            K_stage = stage['K']
            omegas_stage = fourier_frequencies(TY, K_stage)
            path = generate_targets(stage['m_opt'], t_dense, K_stage, omegas_stage)

            lw = 1.0 if s_idx < 2 else 2.0
            ls = '--' if s_idx < 2 else '-'
            alpha = 0.4 if s_idx < 2 else 0.8

            label = f'K={K_stage}' if i == 0 else None
            ax.plot(path[:, 0], path[:, 1], ls, color=stage_colors[s_idx],
                    lw=lw, alpha=alpha, label=label)

        ax.scatter(c0[0], c0[1], s=80, marker='*', c='yellow',
                   edgecolors='black', linewidths=1, zorder=10)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        gain = r['eig_K3'] - r['eig_K1']
        ax.set_title(f'Run {i+1}: EIG={r["eig_K3"]:.1f} (+{gain:.1f})', fontsize=10)
        ax.set_xlabel('x', fontsize=9); ax.set_ylabel('y', fontsize=9)

    if n > 0:
        axes[0, 0].legend(loc='upper right', fontsize=7)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.suptitle('Multi-Refinement: K=1 (red) -> K=2 (orange) -> K=3 (blue)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'{OUTDIR}/refinement_progression_grid.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_refinement_all_K3(mr_data, coords, ic_arr, c0):
    results = mr_data['results']
    n = len(results)
    omegas = fourier_frequencies(TY, K)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                    levels=20, cmap='viridis', alpha=0.6)

    sorted_idx = np.argsort([r['eig_K3'] for r in results])[::-1]
    for rank, i in enumerate(sorted_idx):
        r = results[i]
        path = generate_targets(r['m_opt_final'], t_dense, K, omegas)
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                label=f'Run {i+1}: EIG={r["eig_K3"]:.2f}')

    ax.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
               edgecolors='black', linewidths=1.5, zorder=10, label='c0')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', 'box')
    ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Multi-Refinement: All Final Paths (K=3)', fontsize=13)
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    fname = f'{OUTDIR}/refinement_all_K3_paths.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_comparison_eig_bar_chart(ms_data, mr_data):
    ms_results = ms_data['results']
    mr_results = mr_data['results']
    n = min(len(ms_results), len(mr_results))

    ms_eigs = [ms_results[i]['eig_opt'] for i in range(n)]
    mr_eigs = [mr_results[i]['eig_K3'] for i in range(n)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x_pos = np.arange(n)
    bw = 0.35

    ax.bar(x_pos - bw/2, ms_eigs, bw, alpha=0.8, color='steelblue',
           edgecolor='black', linewidth=0.5, label='Multi-start (direct K=3)')
    ax.bar(x_pos + bw/2, mr_eigs, bw, alpha=0.8, color='coral',
           edgecolor='black', linewidth=0.5, label='Multi-refinement (K=1->2->3)')

    for j in range(n):
        ax.text(j - bw/2, ms_eigs[j] + 0.2, f'{ms_eigs[j]:.1f}',
                ha='center', va='bottom', fontsize=7, color='steelblue')
        ax.text(j + bw/2, mr_eigs[j] + 0.2, f'{mr_eigs[j]:.1f}',
                ha='center', va='bottom', fontsize=7, color='coral')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Run {i+1}' for i in range(n)], fontsize=9)
    ax.set_ylabel('EIG (K=3)', fontsize=12)
    ax.set_title('Multi-Start vs Multi-Refinement: Final EIG Values', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    ms_spread = max(ms_eigs) - min(ms_eigs)
    mr_spread = max(mr_eigs) - min(mr_eigs)
    ax.text(0.98, 0.02,
            f'MS spread: {ms_spread:.2f}\nMR spread: {mr_spread:.2f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = f'{OUTDIR}/comparison_eig_bar_chart.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_comparison_side_by_side(ms_data, mr_data, coords, ic_arr, wind_velocity, mesh, c0):
    ms_results = ms_data['results']
    mr_results = mr_data['results']
    omegas = fourier_frequencies(TY, K)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    n_ms = len(ms_results)
    n_mr = len(mr_results)
    colors_ms = cm.tab10(np.linspace(0, 1, n_ms))
    colors_mr = cm.tab10(np.linspace(0, 1, n_mr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: multi-start
    ax1.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                     levels=20, cmap='viridis', alpha=0.6)
    sorted_ms = np.argsort([r['eig_opt'] for r in ms_results])[::-1]
    for rank, i in enumerate(sorted_ms):
        r = ms_results[i]
        path = generate_targets(r['m_opt'], t_dense, K, omegas)
        ax1.plot(path[:, 0], path[:, 1], '-', color=colors_ms[i], lw=1.5, alpha=0.7,
                 label=f'Run {i+1}: EIG={r["eig_opt"]:.1f}')
    ax1.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
                edgecolors='black', linewidths=1.5, zorder=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x', fontsize=12); ax1.set_ylabel('y', fontsize=12)
    ms_spread = max([r['eig_opt'] for r in ms_results]) - min([r['eig_opt'] for r in ms_results])
    ax1.set_title(f'Multi-Start (direct K=3)\nEIG spread = {ms_spread:.2f}', fontsize=12)
    ax1.legend(loc='upper right', fontsize=6)

    # Right: multi-refinement
    ax2.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                     levels=20, cmap='viridis', alpha=0.6)
    sorted_mr = np.argsort([r['eig_K3'] for r in mr_results])[::-1]
    for rank, i in enumerate(sorted_mr):
        r = mr_results[i]
        path = generate_targets(r['m_opt_final'], t_dense, K, omegas)
        ax2.plot(path[:, 0], path[:, 1], '-', color=colors_mr[i], lw=1.5, alpha=0.7,
                 label=f'Run {i+1}: EIG={r["eig_K3"]:.1f}')
    ax2.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
                edgecolors='black', linewidths=1.5, zorder=10)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('x', fontsize=12); ax2.set_ylabel('y', fontsize=12)
    mr_spread = max([r['eig_K3'] for r in mr_results]) - min([r['eig_K3'] for r in mr_results])
    ax2.set_title(f'Multi-Refinement (K=1->2->3)\nEIG spread = {mr_spread:.2f}', fontsize=12)
    ax2.legend(loc='upper right', fontsize=6)

    plt.suptitle('OED Ill-Posedness: Multi-Start vs Multi-Refinement',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'{OUTDIR}/comparison_side_by_side.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_comparison_spread(ms_data, mr_data):
    ms_results = ms_data['results']
    mr_results = mr_data['results']

    ms_eigs = [r['eig_opt'] for r in ms_results]
    mr_eigs = [r['eig_K3'] for r in mr_results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    data = [ms_eigs, mr_eigs]
    bp = ax.boxplot(data, labels=['Multi-Start\n(direct K=3)', 'Multi-Refinement\n(K=1->2->3)'],
                     patch_artist=True, widths=0.5)

    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.6)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, len(d))
        ax.scatter(x, d, alpha=0.7, s=40, zorder=5,
                   color='steelblue' if i == 0 else 'coral',
                   edgecolors='black', linewidths=0.5)

    ax.set_ylabel('EIG (K=3)', fontsize=12)
    ax.set_title('EIG Distribution: Multi-Start vs Multi-Refinement', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    ms_spread = max(ms_eigs) - min(ms_eigs)
    mr_spread = max(mr_eigs) - min(mr_eigs)
    ratio = ms_spread / mr_spread if mr_spread > 0 else float('inf')
    ax.text(0.98, 0.02,
            f'MS spread: {ms_spread:.2f}\nMR spread: {mr_spread:.2f}\n'
            f'Reduction: {ratio:.1f}x',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = f'{OUTDIR}/comparison_spread.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def main():
    ms_data = load_multi_start('multi_start_results.pkl')
    mr_data = load_multi_refinement('multi_refinement_results.pkl')

    print_comparison_table(ms_data, mr_data)

    print("\nSetting up FE spaces for plotting...")
    mesh, Vh, _ = setup_fe_spaces()

    true_ic = setup_true_initial_condition(Vh)
    coords = Vh.tabulate_dof_coordinates()
    ic_arr = true_ic.get_local()

    wind_coeffs = ms_data['wind_coeffs']
    wind_velocity = reconstruct_wind(wind_coeffs, mesh, ms_data.get('wind_seed'))

    c0 = ms_data['c0']

    print(f"\n  Per-run refinement plots...")
    plot_refinement_per_run(mr_data, coords, ic_arr, c0)

    print(f"\n  Refinement progression grid...")
    plot_refinement_progression_grid(mr_data, coords, ic_arr, c0)

    print(f"\n  All K=3 refinement paths...")
    plot_refinement_all_K3(mr_data, coords, ic_arr, c0)

    print(f"\n  Comparison plots...")
    plot_comparison_eig_bar_chart(ms_data, mr_data)
    plot_comparison_side_by_side(ms_data, mr_data, coords, ic_arr, wind_velocity, mesh, c0)
    plot_comparison_spread(ms_data, mr_data)

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
