"""
Plot results from multi-start OED experiment.

Outputs (all saved to multi_start_plots/):

  PER-RUN (10 individual PNGs):
    run_01_paths.png  ... run_10_paths.png
      Each shows initial (red dashed) + optimal (blue solid) on concentration,
      with sensor positions, drone start, and EIG values.

  SUMMARY (8 PNGs):
    1. initial_guesses.png           — all 10 initial trajectories on concentration
    2. optimal_paths.png             — all 10 optimal paths on concentration
    3. paths_on_wind.png             — all 10 optimal paths on wind field
    4. side_by_side.png              — left: all initials, right: all optimals on wind
    5. individual_runs_grid.png      — 2x5 grid, each cell = init+opt for one run
    6. gradient_convergence.png      — all 10 gradient curves overlaid
    7. eig_bar_chart.png             — initial vs optimal EIG per run
    8. speed_profiles.png            — all 10 speed curves vs v_max
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


def load_results(filename='multi_start_results.pkl'):
    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['results'])} runs")
    print(f"  c0 = ({data['c0'][0]:.2f}, {data['c0'][1]:.2f})")
    print(f"  Wind seed = {data['wind_seed']}")
    return data


def print_summary_table(data):
    results = data['results']
    c0 = data['c0']
    print(f"\n{'=' * 80}")
    print(f"  MULTI-START RESULTS SUMMARY")
    print(f"  c0 = ({c0[0]:.2f}, {c0[1]:.2f}), wind_seed = {data['wind_seed']}")
    print(f"{'=' * 80}")
    print(f"  {'Run':>4s}  {'EIG_init':>9s}  {'EIG_opt':>9s}  {'Gain':>7s}  "
          f"{'|g| init':>10s}  {'|g| final':>10s}  {'Reduction':>10s}  {'Time':>6s}")
    print(f"  {'-' * 72}")
    for r in results:
        g0 = r['grad_norms'][0] if len(r['grad_norms']) > 0 else 0
        gf = r['grad_norms'][-1] if len(r['grad_norms']) > 0 else 0
        g_red = gf / g0 if g0 > 0 else 0
        gain = r['eig_opt'] - r['eig_init']
        print(f"  {r['idx']+1:4d}  {r['eig_init']:9.2f}  {r['eig_opt']:9.2f}  "
              f"{gain:+7.2f}  {g0:10.2e}  {gf:10.2e}  {g_red:10.2e}  "
              f"{r['time']:5.0f}s")
    eig_opts = [r['eig_opt'] for r in results]
    print(f"\n  Best EIG:   {max(eig_opts):.2f} (Run {np.argmax(eig_opts)+1})")
    print(f"  Worst EIG:  {min(eig_opts):.2f} (Run {np.argmin(eig_opts)+1})")
    print(f"  EIG spread: {max(eig_opts) - min(eig_opts):.2f}")
    print(f"  EIG mean:   {np.mean(eig_opts):.2f} +/- {np.std(eig_opts):.2f}")
    print(f"{'=' * 80}")


# ================================================================
#  PER-RUN PLOTS: initial + optimal on concentration
# ================================================================
def plot_per_run(data, coords, ic_arr, c0):
    """Generate one PNG per run: init (red dashed) + opt (blue solid) on concentration."""
    results = data['results']
    omegas = fourier_frequencies(TY, K)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)

    for i, r in enumerate(results):
        path_init = generate_targets(r['m0'], t_dense, K, omegas)
        path_opt = generate_targets(r['m_opt'], t_dense, K, omegas)
        sensors_init = generate_targets(r['m0'], OBSERVATION_TIMES, K, omegas)
        sensors_opt = generate_targets(r['m_opt'], OBSERVATION_TIMES, K, omegas)
        n_s = len(sensors_opt)
        eig_gain = r['eig_opt'] - r['eig_init']

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        # Concentration background
        ct = ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                             levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(ct, ax=ax, fraction=0.046, pad=0.04, label='Concentration')

        # Initial path
        ax.plot(path_init[:, 0], path_init[:, 1], 'r--', lw=2, alpha=0.7,
                label=f'Initial (EIG={r["eig_init"]:.2f})')
        ax.scatter(sensors_init[:, 0], sensors_init[:, 1], c='red', s=30,
                   alpha=0.4, edgecolors='gray', linewidths=0.5, zorder=5)

        # Optimal path
        ax.plot(path_opt[:, 0], path_opt[:, 1], 'b-', lw=2.5, alpha=0.9,
                label=f'Optimized (EIG={r["eig_opt"]:.2f})')
        sc = ax.scatter(sensors_opt[:, 0], sensors_opt[:, 1], c=range(n_s),
                        cmap='coolwarm', s=40, alpha=0.8, edgecolors='black',
                        linewidths=0.8, zorder=5)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.08)
        cb.set_label('Time (red=start, blue=end)', fontsize=9)

        # Drone start
        ax.scatter(c0[0], c0[1], s=150, marker='*', c='yellow',
                   edgecolors='black', linewidths=1.5, zorder=10,
                   label=f'c0=({c0[0]},{c0[1]})')

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
        shape = "circle" if i % 2 == 0 else "ellipse"
        ax.set_title(f'Run {i+1} ({shape} init): EIG {r["eig_init"]:.2f} -> '
                     f'{r["eig_opt"]:.2f} (gain=+{eig_gain:.2f})', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        fname = f'{OUTDIR}/run_{i+1:02d}_paths.png'
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 1: all initial guesses on concentration
# ================================================================
def plot_initial_guesses(data, coords, ic_arr, c0):
    results = data['results']
    omegas = fourier_frequencies(TY, K)
    n = len(results)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                    levels=20, cmap='viridis', alpha=0.6)

    for i, r in enumerate(results):
        path = generate_targets(r['m0'], t_dense, K, omegas)
        shape = "circ" if i % 2 == 0 else "ell"
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                label=f'Init {i+1} ({shape}, EIG={r["eig_init"]:.1f})')

    ax.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
               edgecolors='black', linewidths=1.5, zorder=10, label=f'c0')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', 'box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'All Initial Trajectory Guesses ({n} starts, same c0)', fontsize=13)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    fname = f'{OUTDIR}/initial_guesses.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 2: all optimal paths on concentration
# ================================================================
def plot_optimal_paths(data, coords, ic_arr, c0):
    results = data['results']
    omegas = fourier_frequencies(TY, K)
    n = len(results)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                    levels=20, cmap='viridis', alpha=0.6)

    sorted_idx = np.argsort([r['eig_opt'] for r in results])[::-1]
    for rank, i in enumerate(sorted_idx):
        r = results[i]
        path = generate_targets(r['m_opt'], t_dense, K, omegas)
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                label=f'Run {i+1}: EIG={r["eig_opt"]:.2f}')

    ax.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
               edgecolors='black', linewidths=1.5, zorder=10, label='c0')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', 'box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'All Optimal Trajectories ({n} starts, same wind & c0)', fontsize=13)
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    fname = f'{OUTDIR}/optimal_paths.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 3: all optimal paths on wind field
# ================================================================
def plot_paths_on_wind(data, wind_velocity, mesh, c0):
    results = data['results']
    omegas = fourier_frequencies(TY, K)
    n = len(results)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    mc_q = dl.UnitSquareMesh(15, 15)
    vc_q = dl.interpolate(wind_velocity, dl.VectorFunctionSpace(mc_q, "CG", 1))
    cq = mc_q.coordinates(); vq = vc_q.compute_vertex_values(mc_q)
    nq = len(cq)
    spd = np.sqrt(vq[:nq]**2 + vq[nq:]**2)
    ax.quiver(cq[:, 0], cq[:, 1], vq[:nq], vq[nq:], spd,
              cmap='coolwarm', alpha=0.6, scale=12)

    sorted_idx = np.argsort([r['eig_opt'] for r in results])[::-1]
    for rank, i in enumerate(sorted_idx):
        r = results[i]
        path = generate_targets(r['m_opt'], t_dense, K, omegas)
        ax.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                label=f'Run {i+1}: EIG={r["eig_opt"]:.2f}')

    ax.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
               edgecolors='black', linewidths=1.5, zorder=10, label='c0')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', 'box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Optimal Trajectories on Wind Field', fontsize=13)
    ax.legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    fname = f'{OUTDIR}/paths_on_wind.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 4: side-by-side (initials on conc | optimals on wind)
# ================================================================
def plot_side_by_side(data, coords, ic_arr, wind_velocity, mesh, c0):
    results = data['results']
    omegas = fourier_frequencies(TY, K)
    n = len(results)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: initials on concentration
    ax1.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                     levels=20, cmap='viridis', alpha=0.6)
    for i, r in enumerate(results):
        path = generate_targets(r['m0'], t_dense, K, omegas)
        ax1.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                 label=f'Init {i+1}')
    ax1.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
                edgecolors='black', linewidths=1.5, zorder=10)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_title('Initial Trajectory Guesses', fontsize=13)
    ax1.legend(loc='upper right', fontsize=7, ncol=2)

    # Right: optimals on wind
    mc_q = dl.UnitSquareMesh(15, 15)
    vc_q = dl.interpolate(wind_velocity, dl.VectorFunctionSpace(mc_q, "CG", 1))
    cq = mc_q.coordinates(); vq = vc_q.compute_vertex_values(mc_q)
    nq = len(cq)
    spd = np.sqrt(vq[:nq]**2 + vq[nq:]**2)
    ax2.quiver(cq[:, 0], cq[:, 1], vq[:nq], vq[nq:], spd,
               cmap='coolwarm', alpha=0.6, scale=12)

    sorted_idx = np.argsort([r['eig_opt'] for r in results])[::-1]
    for rank, i in enumerate(sorted_idx):
        r = results[i]
        path = generate_targets(r['m_opt'], t_dense, K, omegas)
        ax2.plot(path[:, 0], path[:, 1], '-', color=colors[i], lw=1.5, alpha=0.7,
                 label=f'Run {i+1}: EIG={r["eig_opt"]:.2f}')
    ax2.scatter(c0[0], c0[1], s=200, marker='*', c='yellow',
                edgecolors='black', linewidths=1.5, zorder=10)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_title('Optimized Trajectories on Wind Field', fontsize=13)
    ax2.legend(loc='upper right', fontsize=7)

    plt.suptitle(f'OED Ill-Posedness: {n} Starts, Same Wind & c0', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fname = f'{OUTDIR}/side_by_side.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 5: 2x5 grid, each cell = init (red) + opt (blue)
# ================================================================
def plot_individual_runs_grid(data, coords, ic_arr, c0):
    results = data['results']
    omegas = fourier_frequencies(TY, K)
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

    for i, r in enumerate(results):
        ax = axes[i // n_cols, i % n_cols]
        ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                        levels=15, cmap='viridis', alpha=0.5)

        path_init = generate_targets(r['m0'], t_dense, K, omegas)
        path_opt = generate_targets(r['m_opt'], t_dense, K, omegas)
        sensors_opt = generate_targets(r['m_opt'], OBSERVATION_TIMES, K, omegas)
        n_s = len(sensors_opt)

        ax.plot(path_init[:, 0], path_init[:, 1], 'r--', lw=1.5, alpha=0.6,
                label='Initial')
        ax.plot(path_opt[:, 0], path_opt[:, 1], 'b-', lw=2, alpha=0.8,
                label='Optimal')
        ax.scatter(sensors_opt[:, 0], sensors_opt[:, 1], c=range(n_s),
                   cmap='coolwarm', s=20, alpha=0.8, edgecolors='black',
                   linewidths=0.5, zorder=5)
        ax.scatter(c0[0], c0[1], s=100, marker='*', c='yellow',
                   edgecolors='black', linewidths=1, zorder=10)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', 'box')
        eig_gain = r['eig_opt'] - r['eig_init']
        ax.set_title(f'Run {i+1}: EIG={r["eig_opt"]:.1f} (+{eig_gain:.1f})',
                     fontsize=10)
        ax.set_xlabel('x', fontsize=9); ax.set_ylabel('y', fontsize=9)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Hide unused
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.suptitle('Individual Runs: Initial (red) vs Optimal (blue)', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fname = f'{OUTDIR}/individual_runs_grid.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 6: gradient convergence (all overlaid)
# ================================================================
def plot_gradient_convergence(data):
    results = data['results']
    n = len(results)
    colors = cm.tab10(np.linspace(0, 1, n))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, r in enumerate(results):
        gn = r['grad_norms']
        if len(gn) > 0:
            ax.semilogy(range(len(gn)), gn, '-', color=colors[i], lw=1.5,
                        alpha=0.8, label=f'Run {i+1} (EIG={r["eig_opt"]:.1f})')

    ax.set_xlabel('Function Evaluation', fontsize=12)
    ax.set_ylabel('||grad J||', fontsize=12)
    ax.set_title('Gradient Norm Convergence - All Starts', fontsize=13)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'{OUTDIR}/gradient_convergence.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 7: EIG bar chart
# ================================================================
def plot_eig_bar_chart(data):
    results = data['results']
    n = len(results)
    colors = cm.tab10(np.linspace(0, 1, n))

    eig_opts = [r['eig_opt'] for r in results]
    eig_inits = [r['eig_init'] for r in results]
    sorted_idx = np.argsort(eig_opts)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x_pos = np.arange(n)
    bw = 0.35

    ax.bar(x_pos - bw / 2, [eig_inits[i] for i in sorted_idx],
           bw, alpha=0.4, color='gray', edgecolor='black', linewidth=0.5,
           label='Initial EIG')
    ax.bar(x_pos + bw / 2, [eig_opts[i] for i in sorted_idx],
           bw, alpha=0.8, color=[colors[i] for i in sorted_idx],
           edgecolor='black', linewidth=0.5, label='Optimal EIG')

    for j, i in enumerate(sorted_idx):
        ax.text(j + bw / 2, eig_opts[i] + 0.2, f'{eig_opts[i]:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Run {i+1}' for i in sorted_idx], fontsize=9)
    ax.set_ylabel('EIG', fontsize=12)
    ax.set_title('EIG Values Across Starts (sorted by optimal EIG)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    spread = max(eig_opts) - min(eig_opts)
    ax.text(0.98, 0.02, f'EIG spread: {spread:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fname = f'{OUTDIR}/eig_bar_chart.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  SUMMARY 8: speed profiles (all overlaid)
# ================================================================
def plot_speed_profiles(data):
    results = data['results']
    n = len(results)
    omegas = fourier_frequencies(TY, K)
    colors = cm.tab10(np.linspace(0, 1, n))
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, r in enumerate(results):
        vx, vy = fourier_velocity(r['m_opt'], t_dense, K, omegas)
        speed = np.sqrt(vx**2 + vy**2)
        ax.plot(t_dense, speed, '-', color=colors[i], lw=1.5, alpha=0.8,
                label=f'Run {i+1} (EIG={r["eig_opt"]:.1f})')

    ax.axhline(V_MAX, color='black', ls='--', lw=2, alpha=0.5,
               label=f'v_max = {V_MAX}')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Speed', fontsize=12)
    ax.set_title('Speed Profiles of Optimal Trajectories', fontsize=13)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, V_MAX * 2.0)
    plt.tight_layout()
    fname = f'{OUTDIR}/speed_profiles.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


# ================================================================
#  MAIN
# ================================================================
def main():
    data = load_results('multi_start_results.pkl')
    print_summary_table(data)

    results = data['results']
    c0 = data['c0']
    wind_coeffs = data['wind_coeffs']

    print("\nSetting up FE spaces...")
    mesh, Vh, _ = setup_fe_spaces()

    true_ic = setup_true_initial_condition(Vh)
    coords = Vh.tabulate_dof_coordinates()
    ic_arr = true_ic.get_local()

    from wind_utils import construct_curved_wind
    if 'pattern' in wind_coeffs:
        wind_velocity = construct_curved_wind(
            mesh, pattern=wind_coeffs['pattern'],
            strength=wind_coeffs['strength'],
            center=wind_coeffs['center'],
            mean_vx=wind_coeffs['mean_vx']
        )
    elif wind_coeffs.get('type') == 'opposing_inlets':
        from wind_utils import compute_velocity_field_opposing_inlets
        wind_velocity = compute_velocity_field_opposing_inlets(
            mesh, speed_left=wind_coeffs['speed_left'],
            speed_top=wind_coeffs['speed_top'],
            Re_val=wind_coeffs['Re']
        )
    elif wind_coeffs.get('type') == 'navier_stokes':
        from wind_utils import compute_velocity_field_navier_stokes
        wind_velocity = compute_velocity_field_navier_stokes(
            mesh, direction=wind_coeffs['direction'],
            speed=wind_coeffs['speed'],
            Re_val=wind_coeffs['Re']
        )
    else:
        wind_velocity, _ = sample_spectral_wind(
            mesh, r_wind=wind_coeffs['r_wind'], sigma=wind_coeffs['sigma'],
            alpha=wind_coeffs['alpha'], mean_vx=wind_coeffs['mean_vx'],
            mean_vy=wind_coeffs['mean_vy'], seed=data['wind_seed']
        )

    # Per-run plots
    print(f"\n  Per-run plots ({len(results)} runs)...")
    plot_per_run(data, coords, ic_arr, c0)

    # Summary plots
    print(f"\n  Summary plots...")
    plot_initial_guesses(data, coords, ic_arr, c0)
    plot_optimal_paths(data, coords, ic_arr, c0)
    plot_paths_on_wind(data, wind_velocity, mesh, c0)
    plot_side_by_side(data, coords, ic_arr, wind_velocity, mesh, c0)
    plot_individual_runs_grid(data, coords, ic_arr, c0)
    plot_gradient_convergence(data)
    plot_eig_bar_chart(data)
    plot_speed_profiles(data)

    print(f"\nAll plots saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
