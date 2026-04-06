"""
Multi-start experiment with HANDCRAFTED initial guesses.

Same wind field + same c0, but 10 geometrically distinct starting shapes:
  1. Small circle (R=0.05)
  2. Medium circle (R=0.12)
  3. Large circle (R=0.25)
  4. Clockwise circle (R=0.12, reversed rotation)
  5. Horizontal ellipse (wide)
  6. Vertical ellipse (tall)
  7. Tilted ellipse 45 degrees
  8. Tilted ellipse 135 degrees
  9. Figure-8 (second Fourier mode active)
  10. Higher modes (modes 2+3 active)
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
from wind_utils import sample_spectral_wind
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
# CONFIGURATION
# ================================================================
N_STARTS = 10
WIND_SEED = 42
FIXED_MEAN_VX = 0.5
C0 = np.array([0.5, 0.5])


LABELS = [
    "Small circle (R=0.05)",
    "Medium circle (R=0.12)",
    "Large circle (R=0.25)",
    "Clockwise circle",
    "Horizontal ellipse",
    "Vertical ellipse",
    "Tilted ellipse 45 deg",
    "Tilted ellipse 135 deg",
    "Figure-8 (mode 2)",
    "Higher modes (2+3)",
]


def create_handcrafted_guesses(c0, K, omegas):
    """
    Create 10 geometrically distinct initial trajectory guesses,
    all centered at c0.
    """
    guesses = []

    for idx in range(10):
        m0 = np.zeros(4 * K + 2)
        m0[0] = c0[0]
        m0[1] = c0[1]

        if idx == 0:
            # Small circle
            R = 0.05
            m0[2] = R; m0[5] = R

        elif idx == 1:
            # Medium circle
            R = 0.12
            m0[2] = R; m0[5] = R

        elif idx == 2:
            # Large circle
            R = 0.25
            m0[2] = R; m0[5] = R

        elif idx == 3:
            # Clockwise circle (flip eta sign)
            R = 0.12
            m0[2] = R; m0[5] = -R

        elif idx == 4:
            # Horizontal ellipse (wide x, narrow y)
            m0[2] = 0.20; m0[5] = 0.06

        elif idx == 5:
            # Vertical ellipse (narrow x, wide y)
            m0[2] = 0.06; m0[5] = 0.20

        elif idx == 6:
            # Tilted ellipse at 45 degrees
            a, b = 0.18, 0.08
            theta = np.pi / 4
            m0[2] = a * np.cos(theta)    # theta_1
            m0[3] = -b * np.sin(theta)   # phi_1
            m0[4] = a * np.sin(theta)    # psi_1
            m0[5] = b * np.cos(theta)    # eta_1

        elif idx == 7:
            # Tilted ellipse at 135 degrees
            a, b = 0.18, 0.08
            theta = 3 * np.pi / 4
            m0[2] = a * np.cos(theta)
            m0[3] = -b * np.sin(theta)
            m0[4] = a * np.sin(theta)
            m0[5] = b * np.cos(theta)

        elif idx == 8:
            # Figure-8: circle + second harmonic
            R = 0.10
            m0[2] = R; m0[5] = R
            m0[6] = 0.08   # theta_2
            m0[9] = 0.08   # eta_2

        elif idx == 9:
            # Higher modes active (modes 2 and 3)
            R = 0.10
            m0[2] = R; m0[5] = R
            m0[7] = 0.06   # phi_2
            m0[8] = 0.06   # psi_2
            m0[10] = 0.04  # theta_3
            m0[13] = 0.04  # eta_3

        # Clip amplitudes to bounds
        for j in range(2, 4 * K + 2):
            m0[j] = np.clip(m0[j], -0.3, 0.3)

        # Shrink if path leaves domain
        t_check = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
        targets_check = generate_targets(m0, t_check, K, omegas)
        for attempt in range(20):
            if targets_check.min() >= 0.05 and targets_check.max() <= 0.95:
                break
            m0[2:] *= 0.9
            targets_check = generate_targets(m0, t_check, K, omegas)

        # Fix x̄, ȳ so that c(t₀) = c0 exactly
        t0_obs = OBSERVATION_TIMES[0]
        shift_x = 0.0
        shift_y = 0.0
        for k in range(K):
            cos_kt = np.cos(omegas[k] * t0_obs)
            sin_kt = np.sin(omegas[k] * t0_obs)
            shift_x += m0[2 + 4*k] * cos_kt + m0[3 + 4*k] * sin_kt
            shift_y += m0[4 + 4*k] * cos_kt + m0[5 + 4*k] * sin_kt
        m0[0] = c0[0] - shift_x
        m0[1] = c0[1] - shift_y

        guesses.append(m0.copy())

    return guesses


def run_single_optimization(idx, m0, c0, mesh, Vh, prior, wind_velocity,
                            K, omegas, r_modes, noise_variance, bounds):
    """Run one OED optimization from initial guess m0."""
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
                 'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
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
        'label': LABELS[idx],
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
    print("  MULTI-START OED: HANDCRAFTED INITIAL GUESSES")
    print("=" * 60)
    print(f"  Wind seed:        {WIND_SEED}")
    print(f"  Mean vx:          {FIXED_MEAN_VX}")
    print(f"  Drone start c0:   ({C0[0]}, {C0[1]})")
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

    # Fixed wind field
    print("\nGenerating fixed wind field...")
    sys.stdout.flush()
    wind_velocity, wind_coeffs = sample_spectral_wind(
        mesh, r_wind=WIND_R, sigma=WIND_SIGMA, alpha=WIND_ALPHA,
        mean_vx=FIXED_MEAN_VX, mean_vy=WIND_MEAN_VY, seed=WIND_SEED
    )
    print(f"  Wind generated (seed={WIND_SEED})")
    sys.stdout.flush()

    # Generate handcrafted initial guesses
    print(f"\nGenerating {N_STARTS} handcrafted initial guesses...")
    sys.stdout.flush()
    initial_guesses = create_handcrafted_guesses(C0, K, omegas)

    for i, m0 in enumerate(initial_guesses):
        print(f"  [{i+1:2d}] {LABELS[i]:30s}  "
              f"theta1={m0[2]:+.3f} phi1={m0[3]:+.3f} "
              f"psi1={m0[4]:+.3f} eta1={m0[5]:+.3f}")
    sys.stdout.flush()

    # Run optimizations
    results = []
    t_start = time.time()

    for i in range(N_STARTS):
        print(f"\n{'='*60}")
        print(f"  START {i+1}/{N_STARTS}: {LABELS[i]}")
        print(f"{'='*60}")
        sys.stdout.flush()

        res = run_single_optimization(
            i, initial_guesses[i], C0, mesh, Vh, prior, wind_velocity,
            K, omegas, R_MODES, NOISE_VARIANCE, BOUNDS
        )
        results.append(res)

        # Save checkpoint
        with open('multi_start_handcrafted_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'wind_seed': WIND_SEED,
                'wind_coeffs': wind_coeffs,
                'mean_vx': FIXED_MEAN_VX,
                'initial_guesses': initial_guesses,
                'labels': LABELS,
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_time/60:.1f} minutes\n")
    print(f"  {'#':>3s}  {'Label':30s}  {'EIG_init':>9s}  {'EIG_opt':>9s}  "
          f"{'Gain':>7s}  {'|g| red':>10s}")
    print(f"  {'-' * 78}")

    eig_opts = []
    for r in results:
        g_red = r['grad_norms'][-1] / r['grad_norms'][0] if len(r['grad_norms']) > 1 else 1.0
        print(f"  {r['idx']+1:3d}  {r['label']:30s}  {r['eig_init']:9.2f}  "
              f"{r['eig_opt']:9.2f}  {r['eig_opt']-r['eig_init']:+7.2f}  {g_red:10.2e}")
        eig_opts.append(r['eig_opt'])

    print(f"\n  Best EIG:   {max(eig_opts):.2f} (Run {np.argmax(eig_opts)+1}: "
          f"{LABELS[np.argmax(eig_opts)]})")
    print(f"  Worst EIG:  {min(eig_opts):.2f} (Run {np.argmin(eig_opts)+1}: "
          f"{LABELS[np.argmin(eig_opts)]})")
    print(f"  EIG spread: {max(eig_opts) - min(eig_opts):.2f}")
    print(f"  EIG mean:   {np.mean(eig_opts):.2f} +/- {np.std(eig_opts):.2f}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
