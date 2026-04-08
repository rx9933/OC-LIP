"""
Training data generation for OED on hIPPYlib buildings mesh.

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
MESH_FILE = '/home/fredkhouri/hippylib/applications/ad_diff/ad_20.xml'

# Buildings
BUILDINGS = [
    {'type': 'rectangle', 'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
    {'type': 'rectangle', 'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]

# Wind velocity priors
WIND_SPEED_LEFT_MEAN = 1.0
WIND_SPEED_LEFT_STD = 0.5
WIND_SPEED_RIGHT_MEAN = 1.0
WIND_SPEED_RIGHT_STD = 0.5

# Fourier perturbation priors (5 modes per wall, decaying amplitude)
N_WALL_MODES = 5
WALL_PERTURB_STDS = [0.6, 0.4, 0.3, 0.2, 0.1]

# Drone position prior
DRONE_POS_MEAN = 0.5
DRONE_POS_STD = 0.15
DRONE_POS_BOUNDS = (0.12, 0.88)

# ================================================================
# PARSE ARGUMENTS
# ================================================================
import argparse

if any('ipykernel' in arg for arg in sys.argv):
    args = argparse.Namespace(job_id=0, n_samples=10, output_prefix='hippylib_training_data')
else:
    parser = argparse.ArgumentParser(description='Generate OED training data (hIPPYlib buildings)')
    parser.add_argument('--job_id', type=int, default=0,
                        help='Job ID for parallel runs')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--output_prefix', type=str, default='hippylib_training_data',
                        help='Prefix for output file')
    args, _ = parser.parse_known_args()


# ================================================================
# SETUP FUNCTIONS
# ================================================================
def v_boundary(x, on_boundary):
    return on_boundary

def q_boundary(x, on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS


def setup_buildings_mesh(speed_left=1.0, speed_right=1.0, coeffs_left=None, coeffs_right=None):
    """Load ad_20 mesh, solve Navier-Stokes with Fourier-perturbed wall BCs."""
    if coeffs_left is None:
        coeffs_left = [0.0] * 5
    if coeffs_right is None:
        coeffs_right = [0.0] * 5

    mesh = dl.refine(dl.Mesh(MESH_FILE))
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)
    g = dl.Expression((
        '0.0',
        '(x[0]<eps) * (bl*4*x[1]*(1-x[1]) + a1l*sin(pi*x[1]) + a2l*sin(2*pi*x[1]) + a3l*sin(3*pi*x[1]) + a4l*sin(4*pi*x[1]) + a5l*sin(5*pi*x[1]))'
        ' + (x[0]>1-eps) * (-br*4*x[1]*(1-x[1]) + a1r*sin(pi*x[1]) + a2r*sin(2*pi*x[1]) + a3r*sin(3*pi*x[1]) + a4r*sin(4*pi*x[1]) + a5r*sin(5*pi*x[1]))'
    ), degree=4, eps=1e-14, pi=np.pi,
       bl=speed_left, br=speed_right,
       a1l=coeffs_left[0], a2l=coeffs_left[1], a3l=coeffs_left[2], a4l=coeffs_left[3], a5l=coeffs_left[4],
       a1r=coeffs_right[0], a2r=coeffs_right[1], a3r=coeffs_right[2], a4r=coeffs_right[3], a5r=coeffs_right[4])

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
    """BiLaplacian prior (same as hIPPYlib tutorial)."""
    prior = BiLaplacianPrior(Vh, GAMMA, DELTA, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.25), Vh).vector()
    return prior


# ================================================================
# SAMPLING FUNCTIONS
# ================================================================
def sample_wind_speeds():
    """Sample baseline wall speeds from Gaussian priors."""
    sl = max(0.1, np.random.normal(WIND_SPEED_LEFT_MEAN, WIND_SPEED_LEFT_STD))
    sr = max(0.1, np.random.normal(WIND_SPEED_RIGHT_MEAN, WIND_SPEED_RIGHT_STD))
    return sl, sr


def sample_wall_perturbations():
    """Sample Fourier perturbation coefficients for both walls."""
    coeffs_left = np.array([np.random.normal(0, s) for s in WALL_PERTURB_STDS])
    coeffs_right = np.array([np.random.normal(0, s) for s in WALL_PERTURB_STDS])
    return coeffs_left, coeffs_right


def point_in_building(x, y, margin_extra=0.02):
    """Check if point is inside any building (with extra margin)."""
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        m = b['margin'] + margin_extra
        if (xmin - m <= x <= xmax + m) and (ymin - m <= y <= ymax + m):
            return True
    return False


def sample_drone_position():
    """Sample drone position avoiding buildings."""
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
# MULTI-REFINEMENT OPTIMIZATION
# ================================================================
def fix_c0_in_m(m, c0, K_stage, omegas):
    """Adjust x_bar, y_bar so c(t0) = c0."""
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
    """Create bounds for L-BFGS-B."""
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
    """Create random K=1 initial guess."""
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
    """Pad solution with zeros for new modes."""
    m_new = np.zeros(4 * K_to + 2)
    m_new[:len(m_opt)] = m_opt.copy()
    return m_new


def objective_with_obstacles(m, c0, Vh, mesh, prior, wind_velocity,
                              stage_K, omegas, eigsolver):
    """OED objective with all penalties including obstacles."""
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


def run_single_stage(stage_K, m0, c0, mesh, Vh, prior, wind_velocity, eigsolver, sample_idx):
    """Run optimization for one K stage."""
    omegas = fourier_frequencies(TY, stage_K)
    bounds = make_bounds(stage_K)
    reset_cached_bbt()

    eval_count = [0]

    def objective(m):
        J, grad, eig_val, pen_val = objective_with_obstacles(
            m, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver
        )
        eval_count[0] += 1
        if eval_count[0] % 5 == 1:
            print(f"      [{sample_idx+1:4d}|K={stage_K}] eval {eval_count[0]:3d}  "
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

    # Final evaluation
    _, _, eig_opt, pen_opt = objective_with_obstacles(
        m_opt, c0, Vh, mesh, prior, wind_velocity, stage_K, omegas, eigsolver
    )

    print(f"      [{sample_idx+1:4d}|K={stage_K}] done: EIG={eig_opt:.2f}  "
          f"pen={pen_opt:.3f}  nfev={result.nfev}")
    sys.stdout.flush()

    return m_opt, eig_opt, pen_opt, result.nfev


def run_multi_refinement(c0, mesh, Vh, prior, wind_velocity, sample_idx):
    """Run K=1 -> K=2 -> K=3 multi-refinement."""
    shared_eigsolver = CachedEigensolver()

    # Stage 1: K=1
    m0_K1 = create_initial_guess_K1(c0)
    m1_opt, eig_K1, pen_K1, nfev_K1 = run_single_stage(
        1, m0_K1, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx
    )

    # Stage 2: K=2
    m0_K2 = pad_solution(m1_opt, 1, 2)
    m2_opt, eig_K2, pen_K2, nfev_K2 = run_single_stage(
        2, m0_K2, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx
    )

    # Stage 3: K=3
    m0_K3 = pad_solution(m2_opt, 2, 3)
    m3_opt, eig_K3, pen_K3, nfev_K3 = run_single_stage(
        3, m0_K3, c0, mesh, Vh, prior, wind_velocity, shared_eigsolver, sample_idx
    )

    total_nfev = nfev_K1 + nfev_K2 + nfev_K3

    return {
        'm_opt': m3_opt.copy(),
        'eig_K1': eig_K1,
        'eig_K2': eig_K2,
        'eig_K3': eig_K3,
        'pen_K3': pen_K3,
        'nfev_total': total_nfev,
    }


# ================================================================
# MAIN TRAINING DATA GENERATION
# ================================================================
def generate_training_data():
    n_samples = args.n_samples
    job_id = args.job_id
    output_file = f"{args.output_prefix}_job{job_id}.pkl"

    print("=" * 70)
    print("  TRAINING DATA GENERATION: HIPPYLIB BUILDINGS EXAMPLE")
    print("=" * 70)
    print(f"  Job ID:          {job_id}")
    print(f"  Samples:         {n_samples}")
    print(f"  Output:          {output_file}")
    print(f"  Wind prior:      sl ~ N({WIND_SPEED_LEFT_MEAN}, {WIND_SPEED_LEFT_STD}^2)")
    print(f"                   sr ~ N({WIND_SPEED_RIGHT_MEAN}, {WIND_SPEED_RIGHT_STD}^2)")
    print(f"  Wall modes:      {N_WALL_MODES} per wall")
    print(f"  Perturb stds:    {WALL_PERTURB_STDS}")
    print(f"  Drone prior:     c0 ~ N({DRONE_POS_MEAN}, {DRONE_POS_STD}^2), avoid buildings")
    print(f"  Multi-refinement: K=1 -> K=2 -> K=3")
    print(f"  R_MODES:         {R_MODES}")
    print(f"  NN input dim:    {2 + 2*N_WALL_MODES + 2} (2 speeds + {2*N_WALL_MODES} perturb + 2 position)")
    print(f"  NN output dim:   {4*K + 2} (Fourier path coefficients)")
    print("=" * 70)
    sys.stdout.flush()

    # Base seed unique per job
    base_seed = int(time.time()) + job_id * 1000000
    print(f"  Base seed: {base_seed}")

    # Check mesh DOFs
    mesh_template = dl.refine(dl.Mesh(MESH_FILE))
    Vh_template = dl.FunctionSpace(mesh_template, "Lagrange", 1)
    n_dofs = Vh_template.dim()
    print(f"  Mesh DOFs: {n_dofs}")
    sys.stdout.flush()

    training_data = []
    successful = 0
    failed = 0
    t_start = time.time()

    for i in range(n_samples):
        seed = base_seed + i
        np.random.seed(seed)

        # Sample wind parameters
        speed_left, speed_right = sample_wind_speeds()
        coeffs_left, coeffs_right = sample_wall_perturbations()

        # Sample drone position
        c0 = sample_drone_position()

        print(f"\n  [{i+1:4d}/{n_samples}] seed={seed}  "
              f"sl={speed_left:.3f}  sr={speed_right:.3f}  "
              f"|cl|={np.linalg.norm(coeffs_left):.3f}  "
              f"|cr|={np.linalg.norm(coeffs_right):.3f}  "
              f"c0=({c0[0]:.3f}, {c0[1]:.3f})")
        sys.stdout.flush()

        t0 = time.time()

        try:
            # Solve Navier-Stokes for this wind
            mesh, Vh, wind_velocity = setup_buildings_mesh(
                speed_left, speed_right, coeffs_left, coeffs_right
            )
            prior = setup_prior_buildings(Vh)

            # Extract full wind field as numpy array (for POD later)
            wind_dof_vector = wind_velocity.vector().get_local().copy()

            # Run multi-refinement
            result = run_multi_refinement(c0, mesh, Vh, prior, wind_velocity, i)

            elapsed = time.time() - t0

            # Build wind_params vector for NN input
            # [speed_left, speed_right, cl_1..cl_5, cr_1..cr_5]
            wind_params = np.concatenate([
                [speed_left, speed_right],
                coeffs_left,
                coeffs_right
            ])

            # Build full NN input: wind_params + drone position
            nn_input = np.concatenate([wind_params, c0])

            # Store sample
            sample = {
                'seed': seed,
                'c_init': c0.copy(),
                'speed_left': speed_left,
                'speed_right': speed_right,
                'coeffs_left': coeffs_left.copy(),
                'coeffs_right': coeffs_right.copy(),
                'wind_params': wind_params.copy(),
                'nn_input': nn_input.copy(),
                'wind_dof_vector': wind_dof_vector,
                'm_opt': result['m_opt'].copy(),
                'eig_K1': result['eig_K1'],
                'eig_K2': result['eig_K2'],
                'eig_K3': result['eig_K3'],
                'pen_K3': result['pen_K3'],
                'nfev_total': result['nfev_total'],
                'time': elapsed,
            }

            training_data.append(sample)
            successful += 1

            gain = result['eig_K3'] - result['eig_K1']
            print(f"    -> EIG: {result['eig_K1']:.2f} -> {result['eig_K2']:.2f} -> {result['eig_K3']:.2f} "
                  f"(+{gain:.2f})  pen={result['pen_K3']:.3f}  "
                  f"nfev={result['nfev_total']}  [{elapsed:.0f}s]")
            sys.stdout.flush()

        except Exception as e:
            elapsed = time.time() - t0
            failed += 1
            print(f"    FAILED: {e}  [{elapsed:.0f}s]")
            sys.stdout.flush()

        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0 or i == n_samples - 1:
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'samples': training_data,
                    'job_id': job_id,
                    'n_samples': len(training_data),
                    'buildings': BUILDINGS,
                    'wind_prior': {
                        'speed_left_mean': WIND_SPEED_LEFT_MEAN,
                        'speed_left_std': WIND_SPEED_LEFT_STD,
                        'speed_right_mean': WIND_SPEED_RIGHT_MEAN,
                        'speed_right_std': WIND_SPEED_RIGHT_STD,
                        'n_wall_modes': N_WALL_MODES,
                        'wall_perturb_stds': WALL_PERTURB_STDS,
                    },
                    'drone_prior': {
                        'mean': DRONE_POS_MEAN,
                        'std': DRONE_POS_STD,
                        'bounds': DRONE_POS_BOUNDS,
                    },
                    'n_dofs': n_dofs,
                    'nn_input_dim': 2 + 2*N_WALL_MODES + 2,
                    'nn_output_dim': 4*K + 2,
                }, f)
            print(f"  (checkpoint: {len(training_data)} samples saved to {output_file})")
            sys.stdout.flush()

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  JOB {job_id} COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Successful: {successful} / {n_samples}")
    print(f"  Failed:     {failed}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Avg time:   {total_time/max(successful,1):.1f}s per sample")

    if successful > 0:
        eigs = [s['eig_K3'] for s in training_data]
        gains = [s['eig_K3'] - s['eig_K1'] for s in training_data]
        print(f"\n  EIG K=3 range:  {min(eigs):.2f} to {max(eigs):.2f}")
        print(f"  EIG K=3 mean:   {np.mean(eigs):.2f} +/- {np.std(eigs):.2f}")
        print(f"  Gain range:     {min(gains):.2f} to {max(gains):.2f}")
        print(f"  Gain mean:      {np.mean(gains):.2f} +/- {np.std(gains):.2f}")

    print(f"\n  Output: {output_file}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    return training_data


if __name__ == "__main__":
    generate_training_data()
