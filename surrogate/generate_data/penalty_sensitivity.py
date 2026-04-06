"""
Penalty sensitivity analysis: measure the cost of each penalty.

Strategy:
  1. Baseline: optimize with ALL penalties
  2. For each penalty, remove JUST that one and re-optimize
  3. Compare EIG values and check constraint violations

This tells us:
  - How much EIG each penalty costs (EIG_no_X - EIG_all)
  - Whether the penalty is actually needed (does removing it cause violations?)
  - Whether the weight is too aggressive (large EIG cost, tiny violation)
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
from fourier_utils import fourier_frequencies, generate_targets, fourier_velocity, get_position_at_time
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior
from training_data_generator import create_initial_guess
from penalties import boundary_penalty_dense, speed_penalty_dense, acceleration_penalty_dense, initial_position_penalty_dense
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
C0 = np.array([0.5, 0.5])
INIT_SEED = 42


def check_constraint_violations(m_opt, c0, K, omegas):
    """
    Check all constraint violations for a given path.
    Returns a dict with violation metrics.
    """
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)
    targets = generate_targets(m_opt, t_dense, K, omegas)
    
    # Boundary violations
    min_x = targets[:, 0].min()
    max_x = targets[:, 0].max()
    min_y = targets[:, 1].min()
    max_y = targets[:, 1].max()
    min_dist_to_boundary = min(min_x, min_y, 1.0 - max_x, 1.0 - max_y)
    
    # Speed violations
    vx, vy = fourier_velocity(m_opt, t_dense, K, omegas)
    speed = np.sqrt(vx**2 + vy**2)
    max_speed = speed.max()
    speed_violation = max(0, max_speed - V_MAX)
    pct_time_over_speed = 100.0 * np.sum(speed > V_MAX) / len(speed)
    
    # Acceleration violations
    from fourier_utils import m_to_xbar_coeffs
    _, coeffs = m_to_xbar_coeffs(m_opt, K)
    ax_arr = np.zeros(len(t_dense))
    ay_arr = np.zeros(len(t_dense))
    for k in range(K):
        w = omegas[k]
        ax_arr += -w**2 * (coeffs[k, 0] * np.cos(w * t_dense) + coeffs[k, 1] * np.sin(w * t_dense))
        ay_arr += -w**2 * (coeffs[k, 2] * np.cos(w * t_dense) + coeffs[k, 3] * np.sin(w * t_dense))
    accel = np.sqrt(ax_arr**2 + ay_arr**2)
    max_accel = accel.max()
    accel_violation = max(0, max_accel - A_MAX)
    pct_time_over_accel = 100.0 * np.sum(accel > A_MAX) / len(accel)
    
    # IC violation
    pos_t0 = get_position_at_time(m_opt, OBSERVATION_TIMES[0], K, omegas)
    ic_distance = np.linalg.norm(pos_t0 - c0)
    
    return {
        'min_dist_to_boundary': min_dist_to_boundary,
        'max_speed': max_speed,
        'speed_violation': speed_violation,
        'pct_time_over_speed': pct_time_over_speed,
        'max_accel': max_accel,
        'accel_violation': accel_violation,
        'pct_time_over_accel': pct_time_over_accel,
        'ic_distance': ic_distance,
    }


def run_optimization(m0, c0, mesh, Vh, prior, wind_velocity, K, omegas,
                     label, enable_bdy=True, enable_spd=True, 
                     enable_acc=True, enable_ic=True):
    """
    Run one optimization with specified penalties enabled/disabled.
    """
    reset_cached_bbt()
    eigsolver = CachedEigensolver()
    
    t0_time = time.time()
    
    # Custom objective that selectively enables penalties
    grad_norms = []
    eig_history = []
    
    def objective(m):
        # Always compute EIG and its gradient
        from oed_objective import build_problem, compute_eig_gradient
        prob, msft, tgts = build_problem(
            m, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            wind_velocity, K, omegas, NOISE_VARIANCE, mesh
        )
        EIG_val, grad_eig, lmbda, V = compute_eig_gradient(
            m, prob, prior, R_MODES, eigsolver, Vh, mesh,
            SIMULATION_TIMES, OBSERVATION_TIMES, K, omegas,
            NOISE_VARIANCE, OBSERVATION_TIMES
        )
        
        grad = -grad_eig.copy()
        pen_val = 0.0
        
        # Selectively add penalties
        if enable_bdy:
            bdy_val, grad_bdy = boundary_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
            pen_val += bdy_val
            grad += grad_bdy
        
        if enable_spd:
            spd_val, grad_spd = speed_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
            pen_val += spd_val
            grad += grad_spd
        
        if enable_acc:
            acc_val, grad_acc = acceleration_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
            pen_val += acc_val
            grad += grad_acc
        
        if enable_ic:
            ic_val, grad_ic = initial_position_penalty_dense(m, OBSERVATION_TIMES, K, omegas, c0)
            pen_val += ic_val
            grad += grad_ic
        
        J = -EIG_val + pen_val
        
        grad_norms.append(np.linalg.norm(grad))
        eig_history.append(EIG_val)
        
        print(f"    [{label:20s}] eval {len(grad_norms):3d}  "
              f"J={J:.4f}  EIG={EIG_val:.4f}  pen={pen_val:.4f}  |g|={grad_norms[-1]:.4e}")
        sys.stdout.flush()
        
        return J, grad
    
    # Compute initial EIG
    J_init, _, = objective(m0)
    eig_init = eig_history[0]
    
    # Optimize
    result = minimize(
        objective, m0,
        jac=True, method='L-BFGS-B', bounds=BOUNDS,
        options={'maxiter': OPT_MAXITER, 'disp': False,
                 'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
    )
    
    m_opt = result.x
    elapsed = time.time() - t0_time
    
    # Compute all penalty values on the final path (even if not used during optimization)
    bdy_val, _ = boundary_penalty_dense(m_opt, OBSERVATION_TIMES, K, omegas)
    spd_val, _ = speed_penalty_dense(m_opt, OBSERVATION_TIMES, K, omegas)
    acc_val, _ = acceleration_penalty_dense(m_opt, OBSERVATION_TIMES, K, omegas)
    ic_val, _ = initial_position_penalty_dense(m_opt, OBSERVATION_TIMES, K, omegas, c0)
    
    # Check constraint violations
    violations = check_constraint_violations(m_opt, c0, K, omegas)
    
    # Report
    if len(grad_norms) > 1:
        print(f"    [{label:20s}] |g|: {grad_norms[0]:.2e} -> {grad_norms[-1]:.2e}")
    print(f"    [{label:20s}] EIG: {eig_init:.2f} -> {eig_history[-1]:.2f}  [{elapsed:.0f}s]")
    sys.stdout.flush()
    
    return {
        'label': label,
        'm0': m0.copy(),
        'm_opt': m_opt.copy(),
        'eig_init': eig_init,
        'eig_opt': eig_history[-1],
        'grad_norms': grad_norms,
        'eig_history': eig_history,
        'converged': result.success,
        'time': elapsed,
        'nit': result.nit,
        'nfev': result.nfev,
        # Penalty values on final path
        'bdy_val': bdy_val,
        'spd_val': spd_val,
        'acc_val': acc_val,
        'ic_val': ic_val,
        # What was enabled
        'enable_bdy': enable_bdy,
        'enable_spd': enable_spd,
        'enable_acc': enable_acc,
        'enable_ic': enable_ic,
        # Constraint violations
        'violations': violations,
    }


def main():
    print("=" * 70)
    print("  PENALTY SENSITIVITY ANALYSIS")
    print("  Remove one penalty at a time, measure EIG cost")
    print("=" * 70)
    print(f"  Wind: Opposing inlets (left=4.0, top=3.0, Re=300)")
    print(f"  c0: ({C0[0]}, {C0[1]})")
    print(f"  Penalty weights: bdy={ZETA_BDY}, spd={ZETA_SPEED}, acc={ZETA_ACCEL}, ic=200")
    print(f"  Constraints: v_max={V_MAX}, a_max={A_MAX}")
    print(f"  R_MODES: {R_MODES}")
    print("=" * 70)
    sys.stdout.flush()
    
    # Setup
    print("\nSetting up FE spaces...")
    sys.stdout.flush()
    mesh, Vh, _ = setup_fe_spaces()
    prior = setup_prior(Vh)
    omegas = fourier_frequencies(TY, K)
    print(f"  Mesh DOFs: {Vh.dim()}")
    sys.stdout.flush()
    
    # Wind field
    from wind_utils import compute_velocity_field_opposing_inlets
    print(f"\nGenerating opposing inlets wind field...")
    sys.stdout.flush()
    wind_velocity = compute_velocity_field_opposing_inlets(
        mesh, speed_left=4.0, speed_top=3.0, Re_val=300.0
    )
    print(f"  Wind generated")
    sys.stdout.flush()
    
    # Same initial guess for all runs
    m0 = create_initial_guess(C0, K, seed=INIT_SEED)
    
    # Define the 5 experiments
    experiments = [
        ("All penalties",     True,  True,  True,  True),
        ("No boundary",       False, True,  True,  True),
        ("No speed",          True,  False, True,  True),
        ("No acceleration",   True,  True,  False, True),
        ("No IC constraint",  True,  True,  True,  False),
    ]
    
    results = []
    t_start = time.time()
    
    for label, bdy, spd, acc, ic in experiments:
        print(f"\n{'='*70}")
        print(f"  {label}")
        disabled = []
        if not bdy: disabled.append("boundary")
        if not spd: disabled.append("speed")
        if not acc: disabled.append("acceleration")
        if not ic: disabled.append("IC")
        if disabled:
            print(f"  Disabled: {', '.join(disabled)}")
        print(f"{'='*70}")
        sys.stdout.flush()
        
        res = run_optimization(
            m0.copy(), C0, mesh, Vh, prior, wind_velocity, K, omegas,
            label, enable_bdy=bdy, enable_spd=spd, enable_acc=acc, enable_ic=ic
        )
        results.append(res)
        
        # Checkpoint
        with open('penalty_sensitivity_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'wind_coeffs': {
                    'type': 'opposing_inlets',
                    'speed_left': 4.0,
                    'speed_top': 3.0,
                    'Re': 300.0
                },
                'm0': m0,
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()
    
    total_time = time.time() - t_start
    
    # Summary table
    baseline = results[0]
    baseline_eig = baseline['eig_opt']
    
    print(f"\n{'=' * 90}")
    print(f"  PENALTY SENSITIVITY RESULTS")
    print(f"{'=' * 90}")
    print(f"  Total time: {total_time/60:.1f} minutes\n")
    
    print(f"  {'Config':22s}  {'EIG':>8s}  {'dEIG':>8s}  "
          f"{'Bdy pen':>8s}  {'Spd pen':>8s}  {'Acc pen':>8s}  {'IC pen':>8s}")
    print(f"  {'-' * 82}")
    
    for r in results:
        deig = r['eig_opt'] - baseline_eig
        print(f"  {r['label']:22s}  {r['eig_opt']:8.2f}  {deig:+8.2f}  "
              f"{r['bdy_val']:8.2f}  {r['spd_val']:8.2f}  {r['acc_val']:8.2f}  {r['ic_val']:8.2f}")
    
    print(f"\n  {'Config':22s}  {'Max spd':>8s}  {'Spd viol':>8s}  {'%t>vmax':>8s}  "
          f"{'Max acc':>8s}  {'Acc viol':>8s}  {'%t>amax':>8s}  "
          f"{'IC dist':>8s}  {'Bdy dist':>8s}")
    print(f"  {'-' * 98}")
    
    for r in results:
        v = r['violations']
        print(f"  {r['label']:22s}  {v['max_speed']:8.3f}  {v['speed_violation']:8.3f}  "
              f"{v['pct_time_over_speed']:7.1f}%  "
              f"{v['max_accel']:8.3f}  {v['accel_violation']:8.3f}  "
              f"{v['pct_time_over_accel']:7.1f}%  "
              f"{v['ic_distance']:8.4f}  {v['min_dist_to_boundary']:8.4f}")
    
    # Interpretation
    print(f"\n  INTERPRETATION:")
    print(f"  {'Penalty':22s}  {'EIG cost':>10s}  {'Verdict':30s}")
    print(f"  {'-' * 66}")
    
    for r in results[1:]:
        deig = r['eig_opt'] - baseline_eig
        v = r['violations']
        
        if r['label'] == "No boundary":
            violated = v['min_dist_to_boundary'] < 0.02
            verdict = "NEEDED - hits wall" if violated else "Weight too high" if deig > 1.0 else "Weight OK"
        elif r['label'] == "No speed":
            violated = v['speed_violation'] > 0.1
            verdict = f"NEEDED - speed={v['max_speed']:.2f}" if violated else "Weight too high" if deig > 1.0 else "Weight OK"
        elif r['label'] == "No acceleration":
            violated = v['accel_violation'] > 0.1
            verdict = f"NEEDED - accel={v['max_accel']:.2f}" if violated else "Weight too high" if deig > 1.0 else "Weight OK"
        elif r['label'] == "No IC constraint":
            violated = v['ic_distance'] > 0.1
            verdict = f"NEEDED - dist={v['ic_distance']:.3f}" if violated else "Weight too high" if deig > 1.0 else "Weight OK"
        else:
            verdict = ""
        
        print(f"  {r['label']:22s}  {deig:+10.2f}  {verdict:30s}")
    
    print(f"{'=' * 90}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
