"""
Multi-start experiment: demonstrate ill-posedness of OED problem.

Fixed wind field + fixed drone starting position c0 + 10 different initial
trajectory guesses -> 10 (potentially different) optimal paths.
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
from fourier_utils import fourier_frequencies, generate_targets, fourier_velocity
from wind_utils import compute_velocity_field_navier_stokes
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior
from training_data_generator import create_initial_guess, create_initial_ellipse
from scipy.optimize import minimize

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

# ================================================================
# CONFIGURATION
# ================================================================
N_STARTS = 10
C0 = np.array([0.5, 0.5])


def create_diverse_initial_guesses(c0, K, omegas, n_starts=10, base_seed=42):
    """
    Create n_starts random initial guesses using the existing
    create_initial_guess and create_initial_ellipse functions,
    all centered at c0 but with different seeds.
    """
    guesses = []
    
    for idx in range(n_starts):
        seed = base_seed + idx
        
        # Alternate between circles and ellipses for diversity
        if idx % 2 == 0:
            m0 = create_initial_guess(c0, K, seed=seed)
        else:
            m0 = create_initial_ellipse(c0, K, seed=seed)
        
        guesses.append(m0)
    
    return guesses


def run_single_optimization(idx, m0, c0, mesh, Vh, prior, wind_velocity,
                            K, omegas, r_modes, noise_variance, bounds):
    """
    Run one OED optimization from initial guess m0.
    """
    reset_cached_bbt()
    eigsolver = CachedEigensolver()
    
    t0 = time.time()
    
    # Compute initial EIG
    _, _, eig_init, _, _, _ = oed_objective_and_grad(
        c0, m0, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, r_modes, noise_variance,
        OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
    )
    
    # Track gradient norms
    grad_norms = []
    eig_history = []
    
    def objective(m):
        J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(
            c0, m, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            wind_velocity, K, omegas, r_modes, noise_variance,
            OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=True
        )
        grad_norms.append(np.linalg.norm(grad))
        eig_history.append(eig_val)
        print(f"    [{idx+1:2d}] eval {len(grad_norms):3d}  "
              f"J={J:.4f}  EIG={eig_val:.4f}  |g|={grad_norms[-1]:.4e}")
        sys.stdout.flush()
        return J, grad
    
    # Run optimizer
    result = minimize(
        objective, m0,
        jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': OPT_MAXITER, 'disp': False,
                 'ftol': OPT_FTOL, 'gtol': 10.0, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
    )
    
    m_opt = result.x
    
    # Compute final EIG without penalties
    eigsolver.reset()
    _, _, eig_opt, _, _, _ = oed_objective_and_grad(
        c0, m_opt, Vh, mesh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, K, omegas, r_modes, noise_variance,
        OBSERVATION_TIMES, eigsolver, obstacles=None, include_penalties=False
    )
    
    elapsed = time.time() - t0
    
    if len(grad_norms) > 1:
        print(f"    [{idx+1:2d}] |g| reduction: {grad_norms[0]:.4e} -> "
              f"{grad_norms[-1]:.4e} (factor {grad_norms[-1]/grad_norms[0]:.2e})")
    print(f"    [{idx+1:2d}] EIG: {eig_init:.2f} -> {eig_opt:.2f}  "
          f"(gain={eig_opt - eig_init:.2f})  [{elapsed:.0f}s]")
    sys.stdout.flush()
    
    return {
        'idx': idx,
        'm0': m0.copy(),
        'm_opt': m_opt.copy(),
        'eig_init': eig_init,
        'eig_opt': eig_opt,
        'grad_norms': grad_norms,
        'eig_history': eig_history,
        'converged': result.success,
        'time': elapsed,
        'nit': result.nit,
        'nfev': result.nfev,
    }


def main():
    print("=" * 60)
    print("  MULTI-START OED EXPERIMENT")
    print("  Navier-Stokes wind, R_MODES=20")
    print("=" * 60)
    print(f"  Wind: Navier-Stokes (east, speed=4.0, Re=100)")
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
    omegas = fourier_frequencies(TY, K)
    print(f"  Mesh DOFs: {Vh.dim()}")
    print(f"  Fourier modes: K={K}, parameter dim = {4*K+2}")
    sys.stdout.flush()
    
    # Fixed wind field — Navier-Stokes (same as PDEOED notebook)
    print(f"\nGenerating Navier-Stokes wind field (east, speed=4.0, Re=100)...")
    sys.stdout.flush()
    from wind_utils import compute_velocity_field_opposing_inlets
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
    
    # Generate initial guesses
    print(f"\nGenerating {N_STARTS} initial trajectory guesses...")
    sys.stdout.flush()
    initial_guesses = create_diverse_initial_guesses(C0, K, omegas, N_STARTS)
    
    for i, m0 in enumerate(initial_guesses):
        shape = "circle" if i % 2 == 0 else "ellipse"
        print(f"  [{i+1:2d}] {shape:8s}  "
              f"theta1={m0[2]:.3f} phi1={m0[3]:.3f} "
              f"psi1={m0[4]:.3f} eta1={m0[5]:.3f}")
    sys.stdout.flush()
    
    # Run optimizations
    results = []
    t_start = time.time()
    
    for i in range(N_STARTS):
        print(f"\n{'='*60}")
        print(f"  START {i+1}/{N_STARTS}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        res = run_single_optimization(
            i, initial_guesses[i], C0, mesh, Vh, prior, wind_velocity,
            K, omegas, R_MODES, NOISE_VARIANCE, BOUNDS
        )
        results.append(res)
        
        # Save checkpoint after each run
        with open('multi_start_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'wind_seed': None,
                'wind_coeffs': wind_coeffs,
                'mean_vx': None,
                'initial_guesses': initial_guesses,
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()
    
    total_time = time.time() - t_start
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time/60:.1f} minutes\n")
    print(f"  {'#':>3s}  {'EIG_init':>9s}  {'EIG_opt':>9s}  "
          f"{'Gain':>7s}  {'|g| red':>10s}")
    print(f"  {'-'*45}")
    
    eig_opts = []
    for r in results:
        g_red = r['grad_norms'][-1] / r['grad_norms'][0] if len(r['grad_norms']) > 1 else 1.0
        print(f"  {r['idx']+1:3d}  {r['eig_init']:9.2f}  "
              f"{r['eig_opt']:9.2f}  {r['eig_opt']-r['eig_init']:+7.2f}  {g_red:10.2e}")
        eig_opts.append(r['eig_opt'])
    
    print(f"\n  EIG range:  {min(eig_opts):.2f} to {max(eig_opts):.2f}")
    print(f"  EIG spread: {max(eig_opts) - min(eig_opts):.2f}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
