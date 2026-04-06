"""
Multi-refinement experiment: solve OED progressively K=1 -> K=2 -> K=3.

Same wind field + same c0 as multi-start experiment.
Run from 10 random initial guesses to show that multi-refinement
consistently finds good solutions (unlike random-start at K=3).

Comparison:
  - multi_start_experiment.py: 10 random inits, directly optimize K=3
  - This script: 10 random inits, optimize K=1 -> pad -> K=2 -> pad -> K=3
"""

import numpy as np
import dolfin as dl
import pickle
import time
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from config import *
from fourier_utils import fourier_frequencies, generate_targets
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# ================================================================
# CONFIGURATION — same wind as multi-start experiment
# ================================================================
N_STARTS = 10
C0 = np.array([0.5, 0.5])


def make_bounds(K_stage):
    """Create bounds array for a given K."""
    lb = np.zeros(4 * K_stage + 2)
    ub = np.zeros(4 * K_stage + 2)
    lb[0] = 0.1; ub[0] = 0.9
    lb[1] = 0.1; ub[1] = 0.9
    for kk in range(K_stage):
        for j in range(4):
            lb[2 + 4 * kk + j] = -0.3
            ub[2 + 4 * kk + j] = 0.3
    return list(zip(lb, ub))


def fix_c0_in_m(m, c0, K_stage, omegas):
    """Adjust x_bar, y_bar so that c(t0) = c0 exactly."""
    t0_obs = OBSERVATION_TIMES[0]
    shift_x = 0.0
    shift_y = 0.0
    for k in range(K_stage):
        cos_kt = np.cos(omegas[k] * t0_obs)
        sin_kt = np.sin(omegas[k] * t0_obs)
        shift_x += m[2 + 4 * k] * cos_kt + m[3 + 4 * k] * sin_kt
        shift_y += m[4 + 4 * k] * cos_kt + m[5 + 4 * k] * sin_kt
    m[0] = c0[0] - shift_x
    m[1] = c0[1] - shift_y
    return m


def create_initial_guess_K1(c0, seed=None):
    """
    Create initial guess for K=1 (6 parameters).
    Random ellipse centered so c(t0) = c0.
    """
    if seed is not None:
        np.random.seed(seed)

    K_stage = 1
    omegas = fourier_frequencies(TY, K_stage)
    m0 = np.zeros(4 * K_stage + 2)

    # Random radius and eccentricity
    radius = np.random.uniform(0.03, 0.25)
    eccentricity = np.random.uniform(0, 0.6)
    a = radius * (1 + eccentricity)
    b = radius * (1 - eccentricity)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    m0[2] = a * np.cos(theta)
    m0[3] = -sign * b * np.sin(theta)
    m0[4] = a * np.sin(theta)
    m0[5] = sign * b * np.cos(theta)

    # Clip
    for j in range(2, 4 * K_stage + 2):
        m0[j] = np.clip(m0[j], -0.3, 0.3)

    # Fix c0
    m0 = fix_c0_in_m(m0, c0, K_stage, omegas)

    return m0


def pad_solution(m_opt, K_from, K_to):
    """
    Pad optimized solution from K_from to K_to by adding zeros
    for the new modes.
    """
    m_new = np.zeros(4 * K_to + 2)
    m_new[:len(m_opt)] = m_opt.copy()
    return m_new


def run_single_stage(stage_K, m0, c0, mesh, Vh, prior, wind_velocity,
                     noise_variance, run_idx, stage_label, eigsolver=None):
    """
    Run optimization for a single K stage.
    """
    omegas = fourier_frequencies(TY, stage_K)
    bounds = make_bounds(stage_K)

    reset_cached_bbt()
    if eigsolver is None:
        eigsolver = CachedEigensolver()

    # Compute initial EIG
    _, _, eig_init, _, _, _ = oed_objective_and_grad(
        c0, m0, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, stage_K, omegas, R_MODES, noise_variance,
        OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
    )

    grad_norms = []
    eig_history = []

    def objective(m):
        J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(
            c0, m, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            wind_velocity, stage_K, omegas, R_MODES, noise_variance,
            OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
        )
        grad_norms.append(np.linalg.norm(grad))
        eig_history.append(eig_val)
        print(f"      [{run_idx+1:2d}|K={stage_K}] eval {len(grad_norms):3d}  "
              f"J={J:.4f}  EIG={eig_val:.4f}  |g|={grad_norms[-1]:.4e}")
        sys.stdout.flush()
        return J, grad

    result = minimize(
        objective, m0,
        jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': OPT_MAXITER, 'disp': False,
                 'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
    )

    m_opt = result.x

    # Compute final J with penalties — use same eigsolver for consistency
    J_opt, _, eig_opt, pen_opt, spd_opt, _ = oed_objective_and_grad(
        c0, m_opt, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, stage_K, omegas, R_MODES, noise_variance,
        OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
    )

    if len(grad_norms) > 1:
        print(f"      [{run_idx+1:2d}|K={stage_K}] |g|: {grad_norms[0]:.2e} -> "
              f"{grad_norms[-1]:.2e} (factor {grad_norms[-1]/grad_norms[0]:.2e})")
    print(f"      [{run_idx+1:2d}|K={stage_K}] EIG: {eig_init:.2f} -> {eig_opt:.2f}  "
          f"J={J_opt:.2f}  pen={pen_opt:.2f}")
    sys.stdout.flush()

    return {
        'K': stage_K,
        'm0': m0.copy(),
        'm_opt': m_opt.copy(),
        'eig_init': eig_init,
        'eig_opt': eig_opt,
        'J_opt': J_opt,
        'pen_opt': pen_opt,
        'spd_opt': spd_opt,
        'grad_norms': grad_norms,
        'eig_history': eig_history,
        'converged': result.success,
        'nit': result.nit,
        'nfev': result.nfev,
    }


def run_multi_refinement(run_idx, c0, mesh, Vh, prior, wind_velocity,
                         noise_variance, seed):
    """
    Run full K=1 -> K=2 -> K=3 refinement for one initial guess.
    """
    t0 = time.time()
    stages = []
    shared_eigsolver = CachedEigensolver()

    # Stage 1: K=1 (6 parameters)
    print(f"\n    Stage 1: K=1 (6 parameters)")
    sys.stdout.flush()
    m0_K1 = create_initial_guess_K1(c0, seed=seed)
    stage1 = run_single_stage(1, m0_K1, c0, mesh, Vh, prior, wind_velocity,
                              noise_variance, run_idx, "K=1", shared_eigsolver)
    stages.append(stage1)

    # Stage 2: K=2 (10 parameters) — pad K=1 solution
    print(f"\n    Stage 2: K=2 (10 parameters) — warm start from K=1")
    sys.stdout.flush()
    m0_K2 = pad_solution(stage1['m_opt'], 1, 2)
    omegas_K2 = fourier_frequencies(TY, 2)
    m0_K2 = fix_c0_in_m(m0_K2, c0, 2, omegas_K2)
    stage2 = run_single_stage(2, m0_K2, c0, mesh, Vh, prior, wind_velocity,
                              noise_variance, run_idx, "K=2", shared_eigsolver)
    stages.append(stage2)

    # Stage 3: K=3 (14 parameters) — pad K=2 solution
    print(f"\n    Stage 3: K=3 (14 parameters) — warm start from K=2")
    sys.stdout.flush()
    m0_K3 = pad_solution(stage2['m_opt'], 2, 3)
    omegas_K3 = fourier_frequencies(TY, 3)
    m0_K3 = fix_c0_in_m(m0_K3, c0, 3, omegas_K3)
    stage3 = run_single_stage(3, m0_K3, c0, mesh, Vh, prior, wind_velocity,
                              noise_variance, run_idx, "K=3", shared_eigsolver)
    stages.append(stage3)

    elapsed = time.time() - t0

    return {
        'idx': run_idx,
        'seed': seed,
        'stages': stages,
        'eig_K1': stage1['eig_opt'],
        'eig_K2': stage2['eig_opt'],
        'eig_K3': stage3['eig_opt'],
        'J_K1': stage1['J_opt'],
        'J_K2': stage2['J_opt'],
        'J_K3': stage3['J_opt'],
        'm_opt_final': stage3['m_opt'].copy(),
        'total_time': elapsed,
    }


def main():
    print("=" * 60)
    print("  MULTI-REFINEMENT OED EXPERIMENT")
    print("  K=1 -> K=2 -> K=3, opposing inlets wind, R_MODES=20")
    print("=" * 60)
    print(f"  Wind: Opposing inlets (left=4.0, top=3.0, Re=300)")
    print(f"  Drone start c0:   ({C0[0]}, {C0[1]})")
    print(f"  R_MODES:          {R_MODES}")
    print(f"  Number of starts: {N_STARTS}")
    print("=" * 60)
    sys.stdout.flush()

    # Setup
    print("\nSetting up FE spaces...")
    sys.stdout.flush()
    mesh, Vh, _ = setup_fe_spaces()
    prior = setup_prior(Vh)
    print(f"  Mesh DOFs: {Vh.dim()}")
    sys.stdout.flush()

    # Fixed wind field — opposing inlets (same as multi-start)
    from wind_utils import compute_velocity_field_opposing_inlets
    print(f"\nGenerating opposing inlets wind field...")
    sys.stdout.flush()
    wind_velocity = compute_velocity_field_opposing_inlets(
        mesh, speed_left=4.0, speed_top=3.0, Re_val=300.0
    )
    wind_coeffs = {
        'type': 'opposing_inlets',
        'speed_left': 4.0,
        'speed_top': 3.0,
        'Re': 300.0
    }
    print(f"  Wind generated (opposing inlets)")
    sys.stdout.flush()

    # Run multi-refinement from 10 different seeds
    results = []
    t_start = time.time()
    base_seed = 42

    for i in range(N_STARTS):
        seed = base_seed + i
        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/{N_STARTS} (seed={seed})")
        print(f"{'='*60}")
        sys.stdout.flush()

        res = run_multi_refinement(i, C0, mesh, Vh, prior, wind_velocity,
                                   NOISE_VARIANCE, seed)
        results.append(res)

        # Checkpoint
        with open('multi_refinement_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'wind_seed': None,
                'wind_coeffs': wind_coeffs,
                'mean_vx': None,
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  MULTI-REFINEMENT RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total time: {total_time/60:.1f} minutes\n")
    print(f"  {'Run':>4s}  {'EIG K=1':>9s}  {'EIG K=2':>9s}  {'EIG K=3':>9s}  "
          f"{'J K=3':>9s}  {'Pen K=3':>9s}  {'Gain 1->3':>10s}  {'Time':>6s}")
    print(f"  {'-' * 75}")

    eig_K3s = []
    J_K3s = []
    for r in results:
        gain = r['eig_K3'] - r['eig_K1']
        # Get penalty from stage 3
        pen_K3 = r['stages'][2]['pen_opt']
        print(f"  {r['idx']+1:4d}  {r['eig_K1']:9.2f}  {r['eig_K2']:9.2f}  "
              f"{r['eig_K3']:9.2f}  {r['J_K3']:9.2f}  {pen_K3:9.2f}  "
              f"{gain:+10.2f}  {r['total_time']:5.0f}s")
        eig_K3s.append(r['eig_K3'])
        J_K3s.append(r['J_K3'])

    print(f"\n  K=3 EIG range:  {min(eig_K3s):.2f} to {max(eig_K3s):.2f}")
    print(f"  K=3 EIG spread: {max(eig_K3s) - min(eig_K3s):.2f}")
    print(f"  K=3 EIG mean:   {np.mean(eig_K3s):.2f} +/- {np.std(eig_K3s):.2f}")
    print(f"  K=3 J range:    {min(J_K3s):.2f} to {max(J_K3s):.2f}")
    print(f"  K=3 J spread:   {max(J_K3s) - min(J_K3s):.2f}")

    # Compare with multi-start if available
    if os.path.exists('multi_start_results.pkl'):
        with open('multi_start_results.pkl', 'rb') as f:
            ms_data = pickle.load(f)
        ms_eigs = [r['eig_opt'] for r in ms_data['results']]
        print(f"\n  --- Comparison with direct K=3 multi-start ---")
        print(f"  Multi-start K=3 spread:      {max(ms_eigs) - min(ms_eigs):.2f}")
        print(f"  Multi-refinement K=3 spread:  {max(eig_K3s) - min(eig_K3s):.2f}")
        print(f"  Multi-start best EIG:         {max(ms_eigs):.2f}")
        print(f"  Multi-refinement best EIG:    {max(eig_K3s):.2f}")
        print(f"  Multi-start mean EIG:         {np.mean(ms_eigs):.2f}")
        print(f"  Multi-refinement mean EIG:    {np.mean(eig_K3s):.2f}")

    print(f"{'=' * 80}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
