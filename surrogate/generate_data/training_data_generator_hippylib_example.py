"""
Training data generation for OED on hIPPYlib buildings mesh.

CHANGES:
  1. NO FOURIER WALL PERTURBATIONS — wind BCs are pure parabolic shear
  2. MULTI-START: 10 random initial guesses per sample, keep best EIG
  3. wind_params is now 2D [speed_left, speed_right]
  4. DIAGNOSTIC PLOT at the end showing wind field, all paths, best selection
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
from fourier_utils import fourier_frequencies, generate_targets
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, build_problem, compute_eig_gradient
from penalties import (boundary_penalty_dense, speed_penalty_dense,
                       acceleration_penalty_dense, initial_position_penalty_dense,
                       obstacle_penalty_dense)
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
MESH_FILE = 'ad_20.xml'

BUILDINGS = [
    {'type': 'rectangle', 'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'type': 'rectangle', 'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]

WIND_SPEED_LEFT_MEAN = 4.0
WIND_SPEED_LEFT_STD = 2.0
WIND_SPEED_RIGHT_MEAN = 4.0
WIND_SPEED_RIGHT_STD = 2.0

DRONE_POS_MEAN = 0.5
DRONE_POS_STD = 0.15
DRONE_POS_BOUNDS = (0.12, 0.88)

# ================================================================
# ARGUMENTS
# ================================================================
import argparse

if any('ipykernel' in arg for arg in sys.argv):
    args = argparse.Namespace(job_id=0, n_samples=10, n_starts=10,
                              output_prefix='hippylib_training_data')
else:
    parser = argparse.ArgumentParser(description='Generate OED training data (hIPPYlib buildings)')
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_starts', type=int, default=10)
    parser.add_argument('--output_prefix', type=str, default='hippylib_training_data')
    args, _ = parser.parse_known_args()


# ================================================================
# SETUP FUNCTIONS
# ================================================================
def v_boundary(x, on_boundary):
    return on_boundary

def q_boundary(x, on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS


def setup_buildings_mesh(speed_left=1.0, speed_right=1.0):
    """Load ad_20 mesh, solve Navier-Stokes with parabolic wall BCs."""
    mesh = dl.refine(dl.Mesh(MESH_FILE))
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)

    g = dl.Expression((
        '0.0',
        '(x[0]<eps) * (bl*4*x[1]*(1-x[1]))'
        ' + (x[0]>1-eps) * (-br*4*x[1]*(1-x[1]))'
    ), degree=4, eps=1e-14,
       bl=speed_left, br=speed_right)

    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')

    vq = dl.Function(XW)
    (v, q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ((2./Re)*ufl.inner(strain(v), strain(v_test))
         + ufl.inner(ufl.nabla_grad(v)*v, v_test)
         - q*ufl.div(v_test)
         + ufl.div(v)*q_test) * ufl.dx

    dl.solve(F == 0, vq, [bc1, bc2],
             solver_parameters={"newton_solver":
                                 {"relative_tolerance": 1e-4,
                                  "maximum_iterations": 100}})

    wind_velocity = dl.project(v, Xh)
    return mesh, Vh, wind_velocity


def setup_prior_buildings(Vh):
    prior = BiLaplacianPrior(Vh, GAMMA, DELTA, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()
    return prior


# ================================================================
# SAMPLING FUNCTIONS
# ================================================================
def sample_wind_speeds():
    sl = max(0.1, np.random.normal(WIND_SPEED_LEFT_MEAN, WIND_SPEED_LEFT_STD))
    sr = max(0.1, np.random.normal(WIND_SPEED_RIGHT_MEAN, WIND_SPEED_RIGHT_STD))
    return sl, sr


def point_in_building(x, y, margin_extra=0.02):
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        m = b['margin'] + margin_extra
        if (xmin - m <= x <= xmax + m) and (ymin - m <= y <= ymax + m):
            return True
    return False


def sample_drone_position():
    max_attempts = 100
    for _ in range(max_attempts):
        x = np.clip(np.random.normal(DRONE_POS_MEAN, DRONE_POS_STD),
                     DRONE_POS_BOUNDS[0], DRONE_POS_BOUNDS[1])
        y = np.clip(np.random.normal(DRONE_POS_MEAN, DRONE_POS_STD),
                     DRONE_POS_BOUNDS[0], DRONE_POS_BOUNDS[1])
        if not point_in_building(x, y):
            return np.array([x, y])
    return np.array([0.5, 0.5])


# ================================================================
# OPTIMIZATION HELPER FUNCTIONS
# ================================================================
def fix_c0_in_m(m, c0, K_stage, omegas):
    t0_obs = OBSERVATION_TIMES[0]
    shift_x = 0.0
    shift_y = 0.0
    for k in range(K_stage):
        cos_kt = np.cos(omegas[k] * t0_obs)
        sin_kt = np.sin(omegas[k] * t0_obs)
        shift_x += m[2 + 4*k] * cos_kt + m[3 + 4*k] * sin_kt
        shift_y += m[4 + 4*k] * cos_kt + m[5 + 4*k] * sin_kt
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
            lb[2 + 4*kk + j] = -0.3
            ub[2 + 4*kk + j] = 0.3
    return list(zip(lb, ub))


def create_initial_guess_K1(c0):
    K_stage = 1
    omegas = fourier_frequencies(TY, K_stage)
    m0 = np.zeros(4 * K_stage + 2)

    radius = np.random.uniform(0.03, 0.12)
    eccentricity = np.random.uniform(0, 0.5)
    a = radius * (1 + eccentricity)
    b = radius * (1 - eccentricity)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])

    m0[2] = a * np.cos(theta)
    m0[3] = -sign * b * np.sin(theta)
    m0[4] = a * np.sin(theta)
    m0[5] = sign * b * np.cos(theta)

    for j in range(2, 6):
        m0[j] = np.clip(m0[j], -0.3, 0.3)

    m0 = fix_c0_in_m(m0, c0, K_stage, omegas)
    return m0


def pad_solution(m_opt, K_from, K_to):
    m_new = np.zeros(4 * K_to + 2)
    m_new[:len(m_opt)] = m_opt.copy()
    return m_new


def objective_with_obstacles(m, c0, Vh, mesh, prior, wind_velocity,
                              stage_K, omegas, eigsolver):
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

    ic_val, grad_ic = initial_position_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas, c0)
    pen_val += ic_val; grad += grad_ic

    obs_val, grad_obs = obstacle_penalty_dense(m, OBSERVATION_TIMES, stage_K, omegas, BUILDINGS)
    pen_val += obs_val; grad += grad_obs

    J = -EIG_val + pen_val
    return J, grad, EIG_val, pen_val


def run_single_stage(stage_K, m0, c0, mesh, Vh, prior, wind_velocity, eigsolver, sample_idx,
                     start_idx=None):
    omegas = fourier_frequencies(TY, stage_K)
    bounds = make_bounds(stage_K)
    reset_cached_bbt()

    eval_count = [0]
    start_str = f"|s{start_idx}" if start_idx is not None else ""

    def objective(m):
        J, grad, eig_val, pen_val = objective_with_obstacles(
            m, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver
        )
        eval_count[0] += 1
        if eval_count[0] % 5 == 1:
            print(f"      [{sample_idx+1:4d}{start_str}|K={stage_K}] eval {eval_count[0]:3d}  "
                  f"J={J:.4f}  EIG={eig_val:.4f}  |g|={np.linalg.norm(grad):.4e}")
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
    _, _, eig_opt, pen_opt = objective_with_obstacles(
        m_opt, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver
    )

    print(f"      [{sample_idx+1:4d}{start_str}|K={stage_K}] done: EIG={eig_opt:.2f}  "
          f"pen={pen_opt:.3f}  nfev={result.nfev}")
    sys.stdout.flush()

    return m_opt, eig_opt, pen_opt, result.nfev


def run_multi_refinement(c0, mesh, Vh, prior, wind_velocity, sample_idx,
                         m0_K1=None, start_idx=None):
    shared_eigsolver = CachedEigensolver()
    omegas_K1 = fourier_frequencies(TY, 1)

    if m0_K1 is None:
        m0_K1 = create_initial_guess_K1(c0)

    prob_initial, _, _ = build_problem(
        m0_K1, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
        wind_velocity, 1, omegas_K1, NOISE_VARIANCE, mesh
    )
    _, _, eig_initial = shared_eigsolver.solve(prob_initial, prior, R_MODES)

    m1_opt, eig_K1, pen_K1, nfev_K1 = run_single_stage(
        1, m0_K1, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx,
        start_idx=start_idx
    )

    m0_K2 = pad_solution(m1_opt, 1, 2)
    m2_opt, eig_K2, pen_K2, nfev_K2 = run_single_stage(
        2, m0_K2, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx,
        start_idx=start_idx
    )

    m0_K3 = pad_solution(m2_opt, 2, 3)
    m3_opt, eig_K3, pen_K3, nfev_K3 = run_single_stage(
        3, m0_K3, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx,
        start_idx=start_idx
    )

    total_nfev = nfev_K1 + nfev_K2 + nfev_K3

    return {
        'm_opt': m3_opt.copy(),
        'm_init': m0_K1.copy(),
        'eig_K0': eig_initial,
        'eig_K1': eig_K1,
        'eig_K2': eig_K2,
        'eig_K3': eig_K3,
        'pen_K3': pen_K3,
        'nfev_total': total_nfev,
    }


# ================================================================
# MULTI-START WRAPPER — now saves ALL paths for plotting
# ================================================================
def run_multi_start_multi_refinement(c0, mesh, Vh, prior, wind_velocity,
                                      sample_idx, n_starts=10):
    best_result = None
    best_eig = -np.inf
    best_idx = -1
    all_eigs = []
    all_m_opts = []  # Save all paths for plotting

    for s in range(n_starts):
        print(f"    Start {s+1}/{n_starts}:")
        sys.stdout.flush()

        try:
            m0_K1 = create_initial_guess_K1(c0)
            result = run_multi_refinement(
                c0, mesh, Vh, prior, wind_velocity, sample_idx,
                m0_K1=m0_K1, start_idx=s
            )

            all_eigs.append(result['eig_K3'])
            all_m_opts.append(result['m_opt'].copy())

            is_best = result['eig_K3'] > best_eig
            if is_best:
                best_eig = result['eig_K3']
                best_result = result
                best_idx = s

            print(f"    Start {s+1}: EIG_K3={result['eig_K3']:.2f} "
                  f"{'*** BEST ***' if is_best else ''}")
            sys.stdout.flush()

        except Exception as e:
            print(f"    Start {s+1}: FAILED ({e})")
            all_eigs.append(np.nan)
            all_m_opts.append(None)
            sys.stdout.flush()

    if best_result is not None:
        best_result['all_eigs'] = np.array(all_eigs)
        best_result['all_m_opts'] = all_m_opts
        best_result['best_start_idx'] = best_idx
        best_result['n_starts'] = n_starts
        best_result['eig_spread'] = np.nanmax(all_eigs) - np.nanmin(all_eigs)

    valid_eigs = [e for e in all_eigs if not np.isnan(e)]
    if len(valid_eigs) > 1:
        print(f"    EIG spread: {min(valid_eigs):.2f} to {max(valid_eigs):.2f} "
              f"(range={max(valid_eigs)-min(valid_eigs):.2f}, "
              f"best={best_eig:.2f})")
    sys.stdout.flush()

    return best_result


# ================================================================
# DIAGNOSTIC PLOTTING
# ================================================================
def plot_multi_start_diagnostic(sample, wind_velocity, mesh, c0, save_path):
    """Plot wind field + all multi-start paths + best highlighted + EIG bar chart."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    observation_times = np.linspace(T_1, T_FINAL, 50)
    omegas = fourier_frequencies(TY, K)

    all_m_opts = sample['all_m_opts']
    all_eigs = sample['all_eigs']
    best_idx = sample['best_start_idx']
    m_best = sample['m_opt']

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ---- Panel 1: Wind field + all paths ----
    ax1 = axes[0]

    # Draw buildings
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        ax1.add_patch(mpatches.Rectangle((xmin, ymin), w, h, color='black', alpha=0.8, zorder=4))
        ax1.add_patch(mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                     color='black', alpha=0.2, linestyle='--', fill=True, zorder=3))

    # Plot wind field
    resolution = 25
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(resolution):
        for j in range(resolution):
            px, py = X[i,j], Y[i,j]
            inside = any(b_['lower'][0] <= px <= b_['upper'][0] and
                        b_['lower'][1] <= py <= b_['upper'][1] for b_ in BUILDINGS)
            if inside:
                U[i,j] = np.nan; V[i,j] = np.nan; continue
            try:
                val = wind_velocity(np.array([px, py]))
                U[i,j] = val[0]; V[i,j] = val[1]
            except:
                U[i,j] = np.nan; V[i,j] = np.nan
    speed = np.sqrt(np.nan_to_num(U)**2 + np.nan_to_num(V)**2)
    mask = ~np.isnan(U)
    ax1.quiver(X[mask], Y[mask], U[mask], V[mask], speed[mask],
               cmap='coolwarm', alpha=0.5, scale=15, width=0.003)

    # Plot all paths (faint)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_m_opts)))
    for s, m_opt in enumerate(all_m_opts):
        if m_opt is None:
            continue
        path = generate_targets(m_opt, observation_times, K, omegas)
        if s == best_idx:
            ax1.plot(path[:, 0], path[:, 1], '-', color='red', linewidth=3,
                    label=f'Best (start {s+1}, EIG={all_eigs[s]:.2f})', zorder=8)
        else:
            ax1.plot(path[:, 0], path[:, 1], '-', color=colors[s], linewidth=1, alpha=0.5,
                    label=f'Start {s+1} (EIG={all_eigs[s]:.2f})')

    ax1.plot(c0[0], c0[1], 'go', markersize=12, label='Start (c0)', zorder=10)
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1]); ax1.set_aspect('equal')
    ax1.set_title(f'Wind + All {len(all_m_opts)} Paths\nsl={sample["speed_left"]:.2f}, sr={sample["speed_right"]:.2f}')
    ax1.legend(fontsize=6, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Best path zoomed with sensor dots ----
    ax2 = axes[1]

    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        ax2.add_patch(mpatches.Rectangle((xmin, ymin), w, h, color='black', alpha=0.8, zorder=4))
        ax2.add_patch(mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                     color='black', alpha=0.2, linestyle='--', fill=True, zorder=3))

    best_path = generate_targets(m_best, observation_times, K, omegas)
    n_obs = len(observation_times)
    ax2.plot(best_path[:, 0], best_path[:, 1], 'r-', linewidth=2.5, label='Best path')
    ax2.scatter(best_path[:, 0], best_path[:, 1], c=range(n_obs),
                cmap='Reds', s=25, alpha=0.7, edgecolors='red', linewidths=0.3, zorder=6)
    ax2.plot(c0[0], c0[1], 'go', markersize=12, label='Start (c0)', zorder=10)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1]); ax2.set_aspect('equal')
    ax2.set_title(f'Best Path (Start {best_idx+1})\n'
                  f'EIG={sample["eig_K3"]:.2f}, pen={sample["pen_K3"]:.4f}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: EIG bar chart for all starts ----
    ax3 = axes[2]

    valid_mask = ~np.isnan(all_eigs)
    bar_colors = ['red' if i == best_idx else 'steelblue' for i in range(len(all_eigs))]
    bars = ax3.bar(range(1, len(all_eigs)+1), all_eigs, color=bar_colors, alpha=0.7)

    for i, (bar, eig) in enumerate(zip(bars, all_eigs)):
        if not np.isnan(eig):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{eig:.1f}', ha='center', va='bottom', fontsize=7)

    ax3.set_xlabel('Start Index')
    ax3.set_ylabel('EIG (K=3)')
    spread = sample['eig_spread']
    ax3.set_title(f'Multi-Start EIG Comparison\nSpread={spread:.2f}, Best=Start {best_idx+1}')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add horizontal lines for reference
    ax3.axhline(y=sample['eig_K0'], color='orange', linestyle='--', linewidth=1,
                label=f'Initial EIG={sample["eig_K0"]:.1f}')
    ax3.axhline(y=np.nanmax(all_eigs), color='red', linestyle=':', linewidth=1,
                label=f'Best={np.nanmax(all_eigs):.1f}')
    ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Diagnostic plot saved: {save_path}")


# ================================================================
# MAIN TRAINING DATA GENERATION
# ================================================================
def generate_training_data():
    n_samples = args.n_samples
    n_starts = args.n_starts
    job_id = args.job_id
    output_file = f"{args.output_prefix}_job{job_id}.pkl"

    print("=" * 70)
    print("  TRAINING DATA GENERATION: HIPPYLIB BUILDINGS EXAMPLE")
    print("  PARABOLIC SHEAR WIND (NO FOURIER PERTURBATIONS) + MULTI-START")
    print("=" * 70)
    print(f"  Job ID:          {job_id}")
    print(f"  Samples:         {n_samples}")
    print(f"  Multi-starts:    {n_starts} per sample")
    print(f"  Output:          {output_file}")
    print(f"  Wind BCs:        parabolic shear (no Fourier perturbations)")
    print(f"  Wind prior:      sl ~ N({WIND_SPEED_LEFT_MEAN}, {WIND_SPEED_LEFT_STD}^2)")
    print(f"                   sr ~ N({WIND_SPEED_RIGHT_MEAN}, {WIND_SPEED_RIGHT_STD}^2)")
    print(f"  Drone prior:     c0 ~ N({DRONE_POS_MEAN}, {DRONE_POS_STD}^2), avoid buildings")
    print(f"  Multi-refinement: K=1 -> K=2 -> K=3 (x{n_starts} starts)")
    print(f"  R_MODES:         {R_MODES}")
    print(f"  Wind params dim: 2 (speed_left, speed_right)")
    print(f"  NN input dim:    4 (2 speeds + 2 position) + POD")
    print(f"  NN output dim:   {4*K + 2} (Fourier path coefficients)")
    print("=" * 70)
    sys.stdout.flush()

    base_seed = int(time.time()) + job_id * 1000000
    print(f"  Base seed: {base_seed}")

    mesh_template = dl.refine(dl.Mesh(MESH_FILE))
    Vh_template = dl.FunctionSpace(mesh_template, "Lagrange", 1)
    n_dofs = Vh_template.dim()
    print(f"  Mesh DOFs: {n_dofs}")
    sys.stdout.flush()

    # Create plot directory
    plot_dir = f"../buildings_plots/job{job_id}"
    os.makedirs(plot_dir, exist_ok=True)

    training_data = []
    successful = 0
    failed = 0
    t_start = time.time()

    for i in range(n_samples):
        seed = base_seed + i
        np.random.seed(seed)

        speed_left, speed_right = sample_wind_speeds()
        c0 = sample_drone_position()

        print(f"\n  [{i+1:4d}/{n_samples}] seed={seed}  "
              f"sl={speed_left:.3f}  sr={speed_right:.3f}  "
              f"c0=({c0[0]:.3f}, {c0[1]:.3f})")
        sys.stdout.flush()

        t0 = time.time()

        try:
            mesh, Vh, wind_velocity = setup_buildings_mesh(speed_left, speed_right)
            prior = setup_prior_buildings(Vh)
            wind_dof_vector = wind_velocity.vector().get_local().copy()

            result = run_multi_start_multi_refinement(
                c0, mesh, Vh, prior, wind_velocity, i, n_starts=n_starts
            )

            if result is None:
                raise RuntimeError("All multi-start attempts failed")

            elapsed = time.time() - t0

            wind_params = np.array([speed_left, speed_right])
            nn_input = np.concatenate([wind_params, c0])

            sample = {
                'seed': seed,
                'c_init': c0.copy(),
                'speed_left': speed_left,
                'speed_right': speed_right,
                'wind_params': wind_params.copy(),
                'nn_input': nn_input.copy(),
                'wind_dof_vector': wind_dof_vector,
                'm_opt': result['m_opt'].copy(),
                'm_init': result['m_init'].copy(),
                'eig_K0': result['eig_K0'],
                'eig_K1': result['eig_K1'],
                'eig_K2': result['eig_K2'],
                'eig_K3': result['eig_K3'],
                'pen_K3': result['pen_K3'],
                'nfev_total': result['nfev_total'],
                'time': elapsed,
                'all_eigs': result.get('all_eigs', np.array([])),
                'all_m_opts': result.get('all_m_opts', []),
                'best_start_idx': result.get('best_start_idx', 0),
                'n_starts': result.get('n_starts', 1),
                'eig_spread': result.get('eig_spread', 0.0),
            }

            training_data.append(sample)
            successful += 1

            gain = result['eig_K3'] - result['eig_K0']
            spread = result.get('eig_spread', 0.0)
            print(f"    -> BEST EIG: {result['eig_K3']:.2f}  "
                  f"(spread={spread:.2f})  "
                  f"gain={gain:.2f}  pen={result['pen_K3']:.3f}  "
                  f"[{elapsed:.0f}s]")
            sys.stdout.flush()

            # Plot diagnostic for first few samples
            if i < 5:
                plot_path = os.path.join(plot_dir, f'sample_{i+1}_multistart.png')
                try:
                    plot_multi_start_diagnostic(sample, wind_velocity, mesh, c0, plot_path)
                except Exception as e:
                    print(f"  Warning: plotting failed: {e}")

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            print(f"    FAILED: {e}  [{elapsed:.0f}s]")
            sys.stdout.flush()

        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0 or i == n_samples - 1:
            # Don't save all_m_opts in checkpoint (too large)
            checkpoint_data = []
            for s in training_data:
                s_copy = {k: v for k, v in s.items() if k != 'all_m_opts'}
                checkpoint_data.append(s_copy)

            with open(output_file, 'wb') as f:
                pickle.dump({
                    'samples': checkpoint_data,
                    'job_id': job_id,
                    'n_samples': len(checkpoint_data),
                    'n_starts': n_starts,
                    'buildings': BUILDINGS,
                    'wind_prior': {
                        'speed_left_mean': WIND_SPEED_LEFT_MEAN,
                        'speed_left_std': WIND_SPEED_LEFT_STD,
                        'speed_right_mean': WIND_SPEED_RIGHT_MEAN,
                        'speed_right_std': WIND_SPEED_RIGHT_STD,
                    },
                    'drone_prior': {
                        'mean': DRONE_POS_MEAN,
                        'std': DRONE_POS_STD,
                        'bounds': DRONE_POS_BOUNDS,
                    },
                    'n_dofs': n_dofs,
                    'nn_input_dim': 4,
                    'nn_output_dim': 4*K + 2,
                }, f)
            print(f"  (checkpoint: {len(checkpoint_data)} samples saved to {output_file})")
            sys.stdout.flush()

    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  JOB {job_id} COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Successful: {successful} / {n_samples}")
    print(f"  Failed:     {failed}")
    print(f"  Multi-starts per sample: {n_starts}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Avg time:   {total_time/max(successful,1):.1f}s per sample")

    if successful > 0:
        eigs = [s['eig_K3'] for s in training_data]
        gains = [s['eig_K3'] - s['eig_K0'] for s in training_data]
        spreads = [s['eig_spread'] for s in training_data]
        print(f"\n  EIG K=3 range:  {min(eigs):.2f} to {max(eigs):.2f}")
        print(f"  EIG K=3 mean:   {np.mean(eigs):.2f} +/- {np.std(eigs):.2f}")
        print(f"  Gain range:     {min(gains):.2f} to {max(gains):.2f}")
        print(f"  Gain mean:      {np.mean(gains):.2f} +/- {np.std(gains):.2f}")
        print(f"  EIG spread mean: {np.mean(spreads):.2f} (avg range across multi-starts)")

    print(f"\n  Output: {output_file}")
    print(f"  Plots:  {plot_dir}/")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    return training_data


if __name__ == "__main__":
    generate_training_data()
