"""
OED experiment on hIPPYlib ad_20 mesh with buildings.

Lid-driven cavity wind around two rectangular buildings.
Runs both multi-start (10 runs) and multi-refinement (10 runs)
on the same wind field for comparison.
"""

import numpy as np
import dolfin as dl
import ufl
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
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from penalties import (boundary_penalty_dense, speed_penalty_dense,
                       acceleration_penalty_dense, initial_position_penalty_dense,
                       obstacle_penalty_dense)
from oed_objective import build_problem, compute_eig_gradient
from training_data_generator import create_initial_guess, create_initial_ellipse
from scipy.optimize import minimize

sys.path.append('../../')
from hippylib import BiLaplacianPrior

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
MESH_FILE = '/home/fredkhouri/hippylib/applications/ad_diff/ad_20.xml'

# Buildings detected from mesh
BUILDINGS = [
    {'type': 'rectangle', 'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'type': 'rectangle', 'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]


# ================================================================
# SETUP FUNCTIONS
# ================================================================
def v_boundary(x, on_boundary):
    return on_boundary

def q_boundary(x, on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS

def setup_buildings_mesh():
    """Load ad_20 mesh with buildings, compute lid-driven cavity wind."""
    mesh = dl.refine(dl.Mesh(MESH_FILE))
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    # Lid-driven cavity wind (same as hIPPYlib tutorial)
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)
    g = dl.Expression(('0.0', '(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)

    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]

    vq = dl.Function(XW)
    (v, q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ((2./Re)*ufl.inner(strain(v), strain(v_test))
         + ufl.inner(ufl.nabla_grad(v)*v, v_test)
         - q*ufl.div(v_test)
         + ufl.div(v)*q_test) * ufl.dx

    dl.solve(F == 0, vq, bcs,
             solver_parameters={"newton_solver":
                                 {"relative_tolerance": 1e-4,
                                  "maximum_iterations": 100}})

    wind_velocity = dl.project(v, Xh)
    return mesh, Vh, wind_velocity


def setup_prior_buildings(Vh):
    """Same prior as hIPPYlib tutorial."""
    prior = BiLaplacianPrior(Vh, GAMMA, DELTA, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()
    return prior


# ================================================================
# CUSTOM OBJECTIVE WITH OBSTACLES
# ================================================================
def buildings_objective_and_grad(c0, m, Vh, mesh, prior, wind_velocity,
                                  K, omegas, eigsolver):
    """
    OED objective with all penalties including obstacle avoidance.
    """
    _t0 = time.time()

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

    # Boundary penalty
    bdy_val, grad_bdy = boundary_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
    pen_val += bdy_val
    grad += grad_bdy

    # Speed penalty
    spd_val, grad_spd = speed_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
    pen_val += spd_val
    grad += grad_spd

    # Acceleration penalty
    acc_val, grad_acc = acceleration_penalty_dense(m, OBSERVATION_TIMES, K, omegas)
    pen_val += acc_val
    grad += grad_acc

    # IC penalty
    if c0 is not None:
        ic_val, grad_ic = initial_position_penalty_dense(
            m, OBSERVATION_TIMES, K, omegas, c0
        )
        pen_val += ic_val
        grad += grad_ic

    # Obstacle penalty (BUILDINGS)
    obs_val, grad_obs = obstacle_penalty_dense(
        m, OBSERVATION_TIMES, K, omegas, BUILDINGS
    )
    pen_val += obs_val
    grad += grad_obs

    J = -EIG_val + pen_val
    _elapsed = time.time() - _t0

    return J, grad, EIG_val, pen_val, obs_val, _elapsed


# ================================================================
# MULTI-START
# ================================================================
def run_multi_start(mesh, Vh, prior, wind_velocity, omegas):
    """Run 10 multi-start optimizations."""
    print(f"\n{'=' * 60}")
    print(f"  MULTI-START (buildings mesh)")
    print(f"{'=' * 60}")

    results = []
    t_start = time.time()
    base_seed = 42

    for i in range(N_STARTS):
        seed = base_seed + i
        print(f"\n{'=' * 60}")
        print(f"  START {i+1}/{N_STARTS} (seed={seed})")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        # Create initial guess
        if i % 2 == 0:
            m0 = create_initial_guess(C0, K, seed=seed)
        else:
            m0 = create_initial_ellipse(C0, K, seed=seed)

        reset_cached_bbt()
        eigsolver = CachedEigensolver()
        t0 = time.time()

        # Initial EIG
        _, _, eig_init, _, _, _ = buildings_objective_and_grad(
            C0, m0, Vh, mesh, prior, wind_velocity, K, omegas, eigsolver
        )

        grad_norms = []
        eig_history = []

        def objective(m):
            J, grad, eig_val, pen_val, obs_val, elapsed = buildings_objective_and_grad(
                C0, m, Vh, mesh, prior, wind_velocity, K, omegas, eigsolver
            )
            grad_norms.append(np.linalg.norm(grad))
            eig_history.append(eig_val)
            print(f"    [{i+1:2d}] eval {len(grad_norms):3d}  "
                  f"J={J:.4f}  EIG={eig_val:.4f}  obs={obs_val:.4f}  |g|={grad_norms[-1]:.4e}")
            sys.stdout.flush()
            return J, grad

        result = minimize(
            objective, m0,
            jac=True, method='L-BFGS-B', bounds=BOUNDS,
            options={'maxiter': OPT_MAXITER, 'disp': False,
                     'ftol': OPT_FTOL, 'gtol': 1e-5,
                     'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
        )

        m_opt = result.x
        print(f"    [{i+1:2d}] Optimizer: {result.message}, nit={result.nit}, nfev={result.nfev}")
        print(f"    [{i+1:2d}] Gradient has NaN: {np.any(np.isnan(result.jac))}")
        elapsed = time.time() - t0

        # Final EIG
        eigsolver.reset()
        _, _, eig_opt, _, _, _ = buildings_objective_and_grad(
            C0, m_opt, Vh, mesh, prior, wind_velocity, K, omegas, eigsolver
        )

        if len(grad_norms) > 1:
            print(f"    [{i+1:2d}] |g| reduction: {grad_norms[0]:.4e} -> "
                  f"{grad_norms[-1]:.4e} (factor {grad_norms[-1]/grad_norms[0]:.2e})")
        print(f"    [{i+1:2d}] EIG: {eig_init:.2f} -> {eig_opt:.2f}  [{elapsed:.0f}s]")
        sys.stdout.flush()

        results.append({
            'idx': i,
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
        })

        # Checkpoint
        with open('buildings_multi_start_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'buildings': BUILDINGS,
                'wind_type': 'lid_driven_cavity',
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()

    total_time = time.time() - t_start
    eig_opts = [r['eig_opt'] for r in results]

    print(f"\n{'=' * 60}")
    print(f"  MULTI-START RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  {'#':>3s}  {'EIG_init':>9s}  {'EIG_opt':>9s}  {'Gain':>7s}")
    print(f"  {'-' * 35}")
    for r in results:
        print(f"  {r['idx']+1:3d}  {r['eig_init']:9.2f}  {r['eig_opt']:9.2f}  "
              f"{r['eig_opt']-r['eig_init']:+7.2f}")
    print(f"\n  EIG range:  {min(eig_opts):.2f} to {max(eig_opts):.2f}")
    print(f"  EIG spread: {max(eig_opts) - min(eig_opts):.2f}")
    print(f"{'=' * 60}")

    return results


# ================================================================
# MULTI-REFINEMENT
# ================================================================
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


def make_bounds(K_stage):
    lb = np.zeros(4 * K_stage + 2)
    ub = np.zeros(4 * K_stage + 2)
    lb[0] = 0.1; ub[0] = 0.9
    lb[1] = 0.1; ub[1] = 0.9
    for kk in range(K_stage):
        for j in range(4):
            lb[2 + 4 * kk + j] = -0.3
            ub[2 + 4 * kk + j] = 0.3
    return list(zip(lb, ub))


def create_initial_guess_K1(c0, seed=None):
    """Create K=1 initial guess."""
    if seed is not None:
        np.random.seed(seed)
    K_stage = 1
    omegas = fourier_frequencies(TY, K_stage)
    m0 = np.zeros(4 * K_stage + 2)

    radius = np.random.uniform(0.03, 0.15)  # Smaller radius for buildings
    eccentricity = np.random.uniform(0, 0.5)
    a = radius * (1 + eccentricity)
    b = radius * (1 - eccentricity)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    m0[2] = a * np.cos(theta)
    m0[3] = -sign * b * np.sin(theta)
    m0[4] = a * np.sin(theta)
    m0[5] = sign * b * np.cos(theta)

    for j in range(2, 4 * K_stage + 2):
        m0[j] = np.clip(m0[j], -0.3, 0.3)

    m0 = fix_c0_in_m(m0, c0, K_stage, omegas)
    return m0


def pad_solution(m_opt, K_from, K_to):
    m_new = np.zeros(4 * K_to + 2)
    m_new[:len(m_opt)] = m_opt.copy()
    return m_new


def run_refinement_stage(stage_K, m0, c0, mesh, Vh, prior, wind_velocity,
                         run_idx, eigsolver):
    """Run one stage of multi-refinement."""
    omegas = fourier_frequencies(TY, stage_K)
    bounds = make_bounds(stage_K)
    reset_cached_bbt()

    # Custom objective for this K
    def stage_objective(m):
        _t0 = time.time()
        prob, msft, tgts = build_problem(
            m, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            wind_velocity, stage_K, omegas, NOISE_VARIANCE, mesh
        )
        EIG_val, grad_eig, lmbda, V = compute_eig_gradient(
            m, prob, prior, R_MODES, eigsolver, Vh, mesh,
            SIMULATION_TIMES, OBSERVATION_TIMES, stage_K, omegas,
            NOISE_VARIANCE, OBSERVATION_TIMES
        )
        grad = -grad_eig.copy()
        pen_val = 0.0

        bdy_val, grad_bdy = boundary_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
        pen_val += bdy_val; grad += grad_bdy

        spd_val, grad_spd = speed_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
        pen_val += spd_val; grad += grad_spd

        acc_val, grad_acc = acceleration_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas)
        pen_val += acc_val; grad += grad_acc

        if c0 is not None:
            ic_val, grad_ic = initial_position_penalty_dense(
                m, OBSERVATION_TIMES, stage_K, omegas, c0
            )
            pen_val += ic_val; grad += grad_ic

        obs_val, grad_obs = obstacle_penalty_dense(
            m, OBSERVATION_TIMES, stage_K, omegas, BUILDINGS
        )
        pen_val += obs_val; grad += grad_obs

        J = -EIG_val + pen_val
        return J, grad, EIG_val, pen_val

    # Initial EIG
    _, _, eig_init, _ = stage_objective(m0)[:4]

    grad_norms = []
    eig_history = []

    def objective(m):
        J, grad, eig_val, pen_val = stage_objective(m)
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
                 'ftol': OPT_FTOL, 'gtol': 1e-5,
                 'maxls': OPT_MAXLS, 'maxfun': OPT_MAXFUN}
    )

    m_opt = result.x

    # Final evaluation
    J_opt, _, eig_opt, pen_opt = stage_objective(m_opt)

    if len(grad_norms) > 1:
        print(f"      [{run_idx+1:2d}|K={stage_K}] |g|: {grad_norms[0]:.2e} -> "
              f"{grad_norms[-1]:.2e}")
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
        'grad_norms': grad_norms,
        'eig_history': eig_history,
        'converged': result.success,
        'nit': result.nit,
        'nfev': result.nfev,
    }


def run_multi_refinement(mesh, Vh, prior, wind_velocity, omegas):
    """Run 10 multi-refinement optimizations (K=1->2->3)."""
    print(f"\n{'=' * 60}")
    print(f"  MULTI-REFINEMENT (buildings mesh)")
    print(f"{'=' * 60}")

    results = []
    t_start = time.time()
    base_seed = 42

    for i in range(N_STARTS):
        seed = base_seed + i
        print(f"\n{'=' * 60}")
        print(f"  RUN {i+1}/{N_STARTS} (seed={seed})")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        t0 = time.time()
        stages = []
        shared_eigsolver = CachedEigensolver()

        # Stage 1: K=1
        print(f"\n    Stage 1: K=1 (6 parameters)")
        sys.stdout.flush()
        m0_K1 = create_initial_guess_K1(C0, seed=seed)
        stage1 = run_refinement_stage(1, m0_K1, C0, mesh, Vh, prior,
                                       wind_velocity, i, shared_eigsolver)
        stages.append(stage1)

        # Stage 2: K=2
        print(f"\n    Stage 2: K=2 (10 parameters)")
        sys.stdout.flush()
        m0_K2 = pad_solution(stage1['m_opt'], 1, 2)
        stage2 = run_refinement_stage(2, m0_K2, C0, mesh, Vh, prior,
                                       wind_velocity, i, shared_eigsolver)
        stages.append(stage2)

        # Stage 3: K=3
        print(f"\n    Stage 3: K=3 (14 parameters)")
        sys.stdout.flush()
        m0_K3 = pad_solution(stage2['m_opt'], 2, 3)
        stage3 = run_refinement_stage(3, m0_K3, C0, mesh, Vh, prior,
                                       wind_velocity, i, shared_eigsolver)
        stages.append(stage3)

        elapsed = time.time() - t0

        results.append({
            'idx': i,
            'seed': seed,
            'stages': stages,
            'eig_K1': stage1['eig_opt'],
            'eig_K2': stage2['eig_opt'],
            'eig_K3': stage3['eig_opt'],
            'm_opt_final': stage3['m_opt'].copy(),
            'total_time': elapsed,
        })

        # Checkpoint
        with open('buildings_multi_refinement_results.pkl', 'wb') as f:
            pickle.dump({
                'results': results,
                'c0': C0,
                'buildings': BUILDINGS,
                'wind_type': 'lid_driven_cavity',
            }, f)
        print(f"  (checkpoint saved)")
        sys.stdout.flush()

    total_time = time.time() - t_start
    eig_K3s = [r['eig_K3'] for r in results]

    print(f"\n{'=' * 60}")
    print(f"  MULTI-REFINEMENT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  {'Run':>4s}  {'EIG K=1':>9s}  {'EIG K=2':>9s}  {'EIG K=3':>9s}  {'Gain':>7s}")
    print(f"  {'-' * 45}")
    for r in results:
        gain = r['eig_K3'] - r['eig_K1']
        print(f"  {r['idx']+1:4d}  {r['eig_K1']:9.2f}  {r['eig_K2']:9.2f}  "
              f"{r['eig_K3']:9.2f}  {gain:+7.2f}")
    print(f"\n  K=3 EIG range:  {min(eig_K3s):.2f} to {max(eig_K3s):.2f}")
    print(f"  K=3 EIG spread: {max(eig_K3s) - min(eig_K3s):.2f}")
    print(f"{'=' * 60}")

    return results


# ================================================================
# PLOTTING
# ================================================================
def plot_buildings_results():
    """Plot results with building obstacles shown."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    omegas = fourier_frequencies(TY, K)
    t_dense = np.linspace(OBSERVATION_TIMES[0], OBSERVATION_TIMES[-1], 200)

    # Load mesh for concentration background
    mesh = dl.refine(dl.Mesh(MESH_FILE))
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    ic_expr = dl.Expression(
        'std::min(0.5, std::exp(-100*(std::pow(x[0]-0.35,2)+std::pow(x[1]-0.7,2))))',
        element=Vh.ufl_element()
    )
    true_ic = dl.interpolate(ic_expr, Vh).vector()
    coords = Vh.tabulate_dof_coordinates()
    ic_arr = true_ic.get_local()

    def draw_buildings(ax):
        for b in BUILDINGS:
            xmin, ymin = b['lower']
            xmax, ymax = b['upper']
            w = xmax - xmin
            h = ymax - ymin
            m = b['margin']
            # Building body
            rect = patches.Rectangle((xmin, ymin), w, h,
                                      color='gray', alpha=0.8, zorder=4)
            ax.add_patch(rect)
            # Margin
            rect_m = patches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                        color='gray', alpha=0.2, linestyle='--',
                                        fill=True, zorder=3)
            ax.add_patch(rect_m)

    os.makedirs('buildings_plots', exist_ok=True)

    # Plot multi-start if available
    if os.path.exists('buildings_multi_start_results.pkl'):
        with open('buildings_multi_start_results.pkl', 'rb') as f:
            ms_data = pickle.load(f)
        ms_results = ms_data['results']
        colors = plt.cm.tab10(np.linspace(0, 1, len(ms_results)))

        # Per-run plots
        for i, r in enumerate(ms_results):
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                            levels=20, cmap='viridis', alpha=0.6)
            draw_buildings(ax)

            # Initial path
            path_init = generate_targets(r['m0'], t_dense, K, omegas)
            ax.plot(path_init[:, 0], path_init[:, 1], 'r--', lw=2, alpha=0.7,
                    label=f'Initial (EIG={r["eig_init"]:.2f})')

            # Optimal path
            path_opt = generate_targets(r['m_opt'], t_dense, K, omegas)
            sensors_opt = generate_targets(r['m_opt'], OBSERVATION_TIMES, K, omegas)
            ax.plot(path_opt[:, 0], path_opt[:, 1], 'b-', lw=2.5, alpha=0.9,
                    label=f'Optimized (EIG={r["eig_opt"]:.2f})')
            n_s = len(sensors_opt)
            ax.scatter(sensors_opt[:, 0], sensors_opt[:, 1], c=range(n_s),
                       cmap='coolwarm', s=30, alpha=0.8, edgecolors='black',
                       linewidths=0.5, zorder=5)

            ax.scatter(C0[0], C0[1], s=150, marker='*', c='yellow',
                       edgecolors='black', linewidths=1.5, zorder=10,
                       label=f'c0=({C0[0]},{C0[1]})')

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_title(f'Run {i+1}: EIG {r["eig_init"]:.2f} -> {r["eig_opt"]:.2f}')
            ax.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.savefig(f'buildings_plots/ms_run_{i+1:02d}.png', dpi=200)
            plt.close()
            print(f"  Saved: buildings_plots/ms_run_{i+1:02d}.png")

        # All optimal paths
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                        levels=20, cmap='viridis', alpha=0.6)
        draw_buildings(ax)

        sorted_idx = np.argsort([r['eig_opt'] for r in ms_results])[::-1]
        for rank, idx in enumerate(sorted_idx):
            r = ms_results[idx]
            path = generate_targets(r['m_opt'], t_dense, K, omegas)
            ax.plot(path[:, 0], path[:, 1], '-', color=colors[idx], lw=1.5, alpha=0.7,
                    label=f'Run {idx+1}: EIG={r["eig_opt"]:.2f}')

        ax.scatter(C0[0], C0[1], s=200, marker='*', c='yellow',
                   edgecolors='black', linewidths=1.5, zorder=10, label='c0')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Multi-Start: All Optimal Paths (buildings)')
        ax.legend(loc='upper left', fontsize=7)
        plt.tight_layout()
        plt.savefig('buildings_plots/ms_all_paths.png', dpi=200)
        plt.close()
        print(f"  Saved: buildings_plots/ms_all_paths.png")

    # Plot multi-refinement if available
   # Plot multi-refinement if available
    if os.path.exists('buildings_multi_refinement_results.pkl'):
        with open('buildings_multi_refinement_results.pkl', 'rb') as f:
            mr_data = pickle.load(f)
        mr_results = mr_data['results']
        colors = plt.cm.tab10(np.linspace(0, 1, len(mr_results)))

        # Per-run refinement plots with time-colored sensors
        for i, r in enumerate(mr_results):
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                            levels=20, cmap='viridis', alpha=0.6)
            draw_buildings(ax)

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

            ax.scatter(C0[0], C0[1], s=150, marker='*', c='yellow',
                       edgecolors='black', linewidths=1.5, zorder=10,
                       label=f'c0=({C0[0]},{C0[1]})')

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            gain = r['eig_K3'] - r['eig_K1']
            ax.set_title(f'Run {i+1}: K=1->K=2->K=3\n'
                         f'EIG: {r["eig_K1"]:.2f} -> {r["eig_K2"]:.2f} -> {r["eig_K3"]:.2f} '
                         f'(+{gain:.2f})')
            ax.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            plt.savefig(f'buildings_plots/mr_run_{i+1:02d}.png', dpi=200)
            plt.close()
            print(f"  Saved: buildings_plots/mr_run_{i+1:02d}.png")

        # All K=3 paths

        # All K=3 paths
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                        levels=20, cmap='viridis', alpha=0.6)
        draw_buildings(ax)

        sorted_idx = np.argsort([r['eig_K3'] for r in mr_results])[::-1]
        for rank, idx in enumerate(sorted_idx):
            r = mr_results[idx]
            path = generate_targets(r['m_opt_final'], t_dense, K, omegas)
            ax.plot(path[:, 0], path[:, 1], '-', color=colors[idx], lw=1.5, alpha=0.7,
                    label=f'Run {idx+1}: EIG={r["eig_K3"]:.2f}')

        ax.scatter(C0[0], C0[1], s=200, marker='*', c='yellow',
                   edgecolors='black', linewidths=1.5, zorder=10, label='c0')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Multi-Refinement: All K=3 Paths (buildings)')
        ax.legend(loc='upper left', fontsize=7)
        plt.tight_layout()
        plt.savefig('buildings_plots/mr_all_K3_paths.png', dpi=200)
        plt.close()
        print(f"  Saved: buildings_plots/mr_all_K3_paths.png")

    # Comparison side-by-side
    if (os.path.exists('buildings_multi_start_results.pkl') and
        os.path.exists('buildings_multi_refinement_results.pkl')):

        with open('buildings_multi_start_results.pkl', 'rb') as f:
            ms_data = pickle.load(f)
        with open('buildings_multi_refinement_results.pkl', 'rb') as f:
            mr_data = pickle.load(f)

        ms_results = ms_data['results']
        mr_results = mr_data['results']
        ms_eigs = [r['eig_opt'] for r in ms_results]
        mr_eigs = [r['eig_K3'] for r in mr_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        colors_ms = plt.cm.tab10(np.linspace(0, 1, len(ms_results)))
        colors_mr = plt.cm.tab10(np.linspace(0, 1, len(mr_results)))

        for ax in [ax1, ax2]:
            ax.tricontourf(coords[:, 0], coords[:, 1], ic_arr,
                            levels=20, cmap='viridis', alpha=0.6)
            draw_buildings(ax)

        # Left: multi-start
        sorted_ms = np.argsort(ms_eigs)[::-1]
        for rank, idx in enumerate(sorted_ms):
            r = ms_results[idx]
            path = generate_targets(r['m_opt'], t_dense, K, omegas)
            ax1.plot(path[:, 0], path[:, 1], '-', color=colors_ms[idx],
                     lw=1.5, alpha=0.7,
                     label=f'Run {idx+1}: EIG={r["eig_opt"]:.1f}')
        ax1.scatter(C0[0], C0[1], s=200, marker='*', c='yellow',
                    edgecolors='black', linewidths=1.5, zorder=10)
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect('equal', 'box')
        ms_spread = max(ms_eigs) - min(ms_eigs)
        ax1.set_title(f'Multi-Start (spread={ms_spread:.2f})')
        ax1.legend(loc='upper left', fontsize=6)

        # Right: multi-refinement
        sorted_mr = np.argsort(mr_eigs)[::-1]
        for rank, idx in enumerate(sorted_mr):
            r = mr_results[idx]
            path = generate_targets(r['m_opt_final'], t_dense, K, omegas)
            ax2.plot(path[:, 0], path[:, 1], '-', color=colors_mr[idx],
                     lw=1.5, alpha=0.7,
                     label=f'Run {idx+1}: EIG={r["eig_K3"]:.1f}')
        ax2.scatter(C0[0], C0[1], s=200, marker='*', c='yellow',
                    edgecolors='black', linewidths=1.5, zorder=10)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal', 'box')
        mr_spread = max(mr_eigs) - min(mr_eigs)
        ax2.set_title(f'Multi-Refinement (spread={mr_spread:.2f})')
        ax2.legend(loc='upper left', fontsize=6)

        plt.suptitle('Buildings Example: Multi-Start vs Multi-Refinement',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('buildings_plots/comparison_side_by_side.png', dpi=200)
        plt.close()
        print(f"  Saved: buildings_plots/comparison_side_by_side.png")


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print("  OED EXPERIMENT: HIPPYLIB BUILDINGS MESH")
    print("=" * 60)
    print(f"  Mesh: {MESH_FILE}")
    print(f"  Buildings: {len(BUILDINGS)} rectangular obstacles")
    for i, b in enumerate(BUILDINGS):
        print(f"    Building {i+1}: x={b['lower'][0]:.2f}-{b['upper'][0]:.2f}, "
              f"y={b['lower'][1]:.2f}-{b['upper'][1]:.2f}, margin={b['margin']}")
    print(f"  c0: ({C0[0]}, {C0[1]})")
    print(f"  R_MODES: {R_MODES}")
    print(f"  N_STARTS: {N_STARTS}")
    print("=" * 60)
    sys.stdout.flush()

    # Setup
    print("\nSetting up buildings mesh and wind field...")
    sys.stdout.flush()
    mesh, Vh, wind_velocity = setup_buildings_mesh()
    prior = setup_prior_buildings(Vh)
    omegas = fourier_frequencies(TY, K)
    print(f"  Mesh DOFs: {Vh.dim()}")
    print(f"  Wind: lid-driven cavity (Re=100)")
    sys.stdout.flush()

    # Run experiments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both',
                        choices=['multi_start', 'multi_refinement', 'both', 'plot'],
                        help='Which experiment to run')
    # Handle Jupyter
    if any('ipykernel' in arg for arg in sys.argv):
        args = argparse.Namespace(mode='both')
    else:
        args = parser.parse_args()

    if args.mode in ['multi_start', 'both']:
        run_multi_start(mesh, Vh, prior, wind_velocity, omegas)

    if args.mode in ['multi_refinement', 'both']:
        run_multi_refinement(mesh, Vh, prior, wind_velocity, omegas)

    if args.mode in ['plot', 'both']:
        print("\nGenerating plots...")
        plot_buildings_results()

    print("\nDone!")


if __name__ == "__main__":
    main()
