"""Training data generation for OED with varying wind and drone positions."""

import numpy as np
import dolfin as dl
import pickle
import time, os
from scipy.optimize import minimize
# Add at the VERY beginning of your script, before importing dolfin
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Try to disable MPI in FEniCS
os.environ['DOLFIN_NOPETSC'] = '1'  # May not work depending on compilation

from config import *
from fourier_utils import fourier_frequencies, xbar_coeffs_to_m
from wind_utils import sample_spectral_wind, coeffs_to_nn_input, nn_input_dim
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior

import warnings
warnings.filterwarnings('ignore')

# Suppress FEniCS logging
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import argparse

import argparse
import sys

# Only parse arguments if not in Jupyter
if any('ipykernel' in arg for arg in sys.argv):
    # We're in Jupyter - use default values
    args = argparse.Namespace(
        job_id=0,
        n_samples=N_SAMPLES,
        output_prefix='oed_training_data'
    )
else:
    # We're in command line - parse arguments
    parser = argparse.ArgumentParser(description='Generate OED training data')
    parser.add_argument('--job_id', type=int, default=0, 
                        help='Job ID for parallel runs (used as seed offset)')
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES,
                        help='Number of samples to generate in this job')
    parser.add_argument('--output_prefix', type=str, default='oed_training_data',
                        help='Prefix for output file')
    args = parser.parse_args()

def sample_mean_vx(mean=MEAN_VX_MEAN, std=MEAN_VX_STD):
    """
    Sample mean_vx from Gaussian distribution.
    
    Parameters
    ----------
    mean : float
        Mean of distribution
    std : float
        Standard deviation of distribution
        
    Returns
    -------
    float
        Sampled mean_vx
    """
    return np.random.normal(mean, std)


def sample_drone_position(mean=DRONE_POS_MEAN, std=DRONE_POS_STD, bounds=(0.1, 0.9)):
    """
    Sample drone starting position from Gaussian, clipped to bounds.
    
    Parameters
    ----------
    mean : float
        Mean of distribution (same for x and y)
    std : float
        Standard deviation
    bounds : tuple
        (lower, upper) bounds for position
        
    Returns
    -------
    np.ndarray
        (2,) array of [x, y]
    """
    x = np.random.normal(mean, std)
    y = np.random.normal(mean, std)
    # Clip to bounds
    x = np.clip(x, bounds[0], bounds[1])
    y = np.clip(y, bounds[0], bounds[1])
    return np.array([x, y])


def create_initial_guess(c_init, K, radius_mean=0.1, radius_std=0.03, seed=None):
    """
    Create initial Fourier parameters that generate a circular path around the mean.
    The radius follows a Gaussian distribution, and the path is constrained to stay
    within the domain boundaries.
    
    Parameters
    ----------
    c_init : np.ndarray
        (2,) initial drone position (mean position of circle)
    K : int
        Number of Fourier modes (only first mode used for circle)
    radius_mean : float
        Mean radius of the circle
    radius_std : float
        Standard deviation of radius distribution
    seed : int or None
        Random seed
        
    Returns
    -------
    np.ndarray
        Flat Fourier parameter vector that generates a circular path
    """
    if seed is not None:
        np.random.seed(seed)
    
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]  # x̄
    m0[1] = c_init[1]  # ȳ
    
    # Sample radius from Gaussian (ensuring it's positive)
    radius = abs(np.random.normal(radius_mean, radius_std))
    
    # For a perfect circle with period T, we need:
    # x(t) = x̄ + R cos(ωt)
    # y(t) = ȳ + R sin(ωt)
    # This corresponds to Fourier coefficients:
    # θ₁ = R, φ₁ = 0, ψ₁ = 0, η₁ = R
    # (with ω = 2π/T for the first mode)
    
    # First mode creates the circle
    m0[2] = radius      # θ₁ - cosine coefficient for x
    m0[3] = 0.0         # φ₁ - sine coefficient for x (zero for circle)
    m0[4] = 0.0         # ψ₁ - cosine coefficient for y (zero for circle)
    m0[5] = radius      # η₁ - sine coefficient for y
    
    # Check if the circle would go out of bounds and adjust if needed
    x_min = m0[0] - radius
    x_max = m0[0] + radius
    y_min = m0[1] - radius
    y_max = m0[1] + radius
    
    # If out of bounds, scale down the radius
    bounds = [0.1, 0.9]  # Domain boundaries with margin
    
    if x_min < bounds[0] or x_max > bounds[1] or y_min < bounds[0] or y_max > bounds[1]:
        # Calculate maximum allowed radius
        max_radius_x = min(m0[0] - bounds[0], bounds[1] - m0[0])
        max_radius_y = min(m0[1] - bounds[0], bounds[1] - m0[1])
        max_radius = min(max_radius_x, max_radius_y, radius)
        
        if max_radius < radius:
            # Rescale
            scale_factor = max_radius / radius
            m0[2] = radius * scale_factor
            m0[5] = radius * scale_factor
            print(f"  Radius scaled from {radius:.3f} to {radius*scale_factor:.3f} to fit in bounds")
    
    # Higher modes get small random perturbations (optional)
    for k in range(1, K):
        # Much smaller amplitudes for higher modes to maintain approximate circle
        scale = radius * 0.1 / (k + 1)  # Decaying with mode number
        m0[2 + 4*k] = np.random.normal(0, scale)      # θ_k
        m0[3 + 4*k] = np.random.normal(0, scale)      # φ_k
        m0[4 + 4*k] = np.random.normal(0, scale)      # ψ_k
        m0[5 + 4*k] = np.random.normal(0, scale)      # η_k
    
    return m0


# Alternative version that creates an ellipse with random orientation
def create_initial_ellipse(c_init, K, radius_mean=0.1, radius_std=0.03, eccentricity_std=0.2, seed=None):
    """
    Create an elliptical path with random orientation and eccentricity.
    """
    if seed is not None:
        np.random.seed(seed)
    
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]
    m0[1] = c_init[1]
    
    # Sample mean radius from Gaussian
    R = abs(np.random.normal(radius_mean, radius_std))
    
    # Add eccentricity (a = R + e, b = R - e)
    e = abs(np.random.normal(0, eccentricity_std))
    a = max(0.01, R + e)
    b = max(0.01, R - e)
    
    # Random orientation angle
    theta = np.random.uniform(0, np.pi)
    
    # For an ellipse rotated by angle theta:
    # x(t) = x̄ + a cos(θ) cos(ωt) - b sin(θ) sin(ωt)
    # y(t) = ȳ + a sin(θ) cos(ωt) + b cos(θ) sin(ωt)
    
    m0[2] = a * np.cos(theta)      # θ₁
    m0[3] = -b * np.sin(theta)      # φ₁
    m0[4] = a * np.sin(theta)      # ψ₁
    m0[5] = b * np.cos(theta)      # η₁
    
    # Check bounds and scale if needed
    # (simplified check - worst-case extent in each direction)
    max_extent_x = abs(m0[2]) + abs(m0[3])
    max_extent_y = abs(m0[4]) + abs(m0[5])
    
    bounds = [0.1, 0.9]
    if (m0[0] - max_extent_x < bounds[0] or m0[0] + max_extent_x > bounds[1] or
        m0[1] - max_extent_y < bounds[0] or m0[1] + max_extent_y > bounds[1]):
        
        # Scale down to fit
        scale_x = min(m0[0] - bounds[0], bounds[1] - m0[0]) / max_extent_x
        scale_y = min(m0[1] - bounds[0], bounds[1] - m0[1]) / max_extent_y
        scale = min(scale_x, scale_y, 1.0)
        
        m0[2:6] *= scale
        print(f"  Ellipse scaled by {scale:.3f} to fit in bounds")
    
    return m0


def generate_training_sample(seed, mesh, Vh, prior, simulation_times,
                             observation_times, t_param, K, omegas,
                             r_modes, noise_variance, bounds, wind_params=None):
    """
    Generate a single training sample.
    
    Parameters
    ----------
    seed : int
        Random seed
    mesh : dolfin Mesh
        Computational mesh
    Vh : dolfin FunctionSpace
        Finite element space
    prior : BiLaplacianPrior
        Prior distribution
    simulation_times : np.ndarray
        Times for simulation
    observation_times : np.ndarray
        Times for observations
    t_param : np.ndarray
        Observation times (usually same as observation_times)
    K : int
        Number of Fourier modes
    omegas : np.ndarray
        Angular frequencies
    r_modes : int
        Number of eigenvalues to keep
    noise_variance : float
        Noise variance for observations
    bounds : list
        List of (lb, ub) tuples for optimization
    wind_params : dict or None
        Wind parameters (if None, use defaults)
        
    Returns
    -------
    tuple
        (sample_dict, success_bool)
    """
    if wind_params is None:
        wind_params = {
            'r_wind': WIND_R,
            'sigma': WIND_SIGMA,
            'alpha': WIND_ALPHA,
            'mean_vx': sample_mean_vx(),
            'mean_vy': WIND_MEAN_VY
        }
    
    # Sample drone position
    c_init = sample_drone_position()
    
    # Sample wind field
    wind_velocity, wind_coeffs = sample_spectral_wind(
        mesh, 
        r_wind=wind_params['r_wind'],
        sigma=wind_params['sigma'],
        alpha=wind_params['alpha'],
        mean_vx=wind_params['mean_vx'],
        mean_vy=wind_params['mean_vy'],
        seed=seed
    )
    
    # Create initial guess
    m0 = create_initial_guess(c_init, K)
    
    # Reset solver state
    reset_cached_bbt()
    eigsolver = CachedEigensolver()
    
    t0 = time.time()
    
    try:
        # ===== ADD THIS SECTION TO COMPUTE INITIAL EIG =====
        print(f"    Computing initial EIG...")
        # Compute initial EIG (without penalties)
        _, _, eig_init, _, _, _ = oed_objective_and_grad(c_init,
            m0, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=True 
        )
        print(f"    Initial EIG = {eig_init:.2f}")
        # ===== END OF ADDED SECTION =====
        sys.stdout.flush() 
        # Define objective function for this sample
        def objective(m):
            J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(c_init,
                m, Vh, mesh, prior, simulation_times, observation_times,
                wind_velocity, K, omegas, r_modes, noise_variance, t_param,
                eigsolver, obstacles=None, include_penalties=True
            )
            
            sys.stdout.flush()

            return J, grad
        
        # Optimize
        result = minimize(
            objective, m0,
            jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': OPT_MAXITER, 'disp': True,
                     'ftol': OPT_FTOL, 'maxls': OPT_MAXLS, 'maxfun':OPT_MAXFUN}
        )
        
        m_opt = result.x
        
        # Compute final EIG (without penalties)
        eigsolver.reset()
        _, _, eig_opt, _, _, _ = oed_objective_and_grad(c_init,
            m_opt, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=False
        )
        
        elapsed = time.time() - t0
        
        # NN input and output
        nn_input = coeffs_to_nn_input(wind_coeffs, c_init)
        nn_output = m_opt.copy()
        
        # ===== MODIFIED SAMPLE DICTIONARY TO INCLUDE eig_init =====
        sample = {
            'seed': seed,
            'c_init': c_init.copy(),
            'mean_vx': wind_params['mean_vx'],
            'nn_input': nn_input,
            'nn_output': nn_output,
            'eig_init': eig_init,  # Added initial EIG
            'eig_opt': eig_opt,
            'eig_gain': eig_opt - eig_init,  # Optional: add gain
            'converged': result.success,
            'wind_coeffs': wind_coeffs,
            'time': elapsed,
            'nit': result.nit,
            'nfev': result.nfev,
        }
        # ===== END OF MODIFIED SECTION =====
        
        return sample, result.success
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED: {e}")
        return None, False

def generate_training_data(n_samples=args.n_samples, 
                          output_file=f"{args.output_prefix}_job{args.job_id}.pkl"):
    """
    Generate training data for OED.
    """
    print("="*60)
    print("  OED TRAINING DATA GENERATION")
    print("="*60)
    print(f"  Job ID: {args.job_id}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Mean_vx distribution: N({MEAN_VX_MEAN}, {MEAN_VX_STD}²)")
    print(f"  Drone position distribution: N({DRONE_POS_MEAN}, {DRONE_POS_STD}²)")
    print("="*60)
    
    # Use job_id as part of the base seed to ensure unique seeds across jobs
    base_seed = int(time.time()) + args.job_id * 1000000
    print(f"Using base seed: {base_seed} (based on current time + job_id offset)")
    max_attempts = n_samples * 3

    # Setup - ADD DEBUGGING HERE
    print("Calling setup_fe_spaces()...")
    import sys
    sys.stdout.flush()  # Force print to appear immediately
    
    mesh, Vh, wind_velocity = setup_fe_spaces()
    print("setup_fe_spaces() completed successfully")
    sys.stdout.flush()
    
    print("Calling setup_prior()...")
    sys.stdout.flush()
    prior = setup_prior(Vh)
    print("setup_prior() completed successfully")
    sys.stdout.flush()
    
    # Compute frequencies
    print("Computing frequencies...")
    omegas = fourier_frequencies(TY, K)
    print(f"Frequencies computed: {omegas}")
    sys.stdout.flush()

    print(f"\n  Mesh DOFs: {Vh.dim()}")
    print(f"  Fourier modes: {K} (parameter dim: {4*K + 2})")
    print(f"  NN input dim: {nn_input_dim(WIND_R)}")
    sys.stdout.flush()


    training_data = []
    total = n_samples
    count = 0
    successful = 0
    t_start_all = time.time()
    total_attempts = 0
    
    while successful < n_samples and total_attempts < max_attempts:
        total_attempts += 1
        
        # Generate seed for this attempt
        seed = base_seed + total_attempts
        np.random.seed(seed)
        count += 1
        
        # Sample mean_vx
        mean_vx = sample_mean_vx()
        
        wind_params = {
            'r_wind': WIND_R,
            'sigma': WIND_SIGMA,
            'alpha': WIND_ALPHA,
            'mean_vx': mean_vx,
            'mean_vy': WIND_MEAN_VY
        }
        
        print(f"\n  [{count}/{total}] seed={seed:3d} mean_vx={mean_vx:.3f}")
        sys.stdout.flush()

        sample, success = generate_training_sample(
            seed, mesh, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            OBSERVATION_TIMES, K, omegas, R_MODES, NOISE_VARIANCE,
            BOUNDS, wind_params
        )
        
        if success:
            training_data.append(sample)
            successful += 1
            print(f"    → EIG: {sample['eig_init']:.2f} → {sample['eig_opt']:.2f} "
                    f"(gain={sample['eig_gain']:.2f}) conv={sample['converged']} "
                    f"[{sample['time']:.1f}s]")
            sys.stdout.flush()
            with open(output_file, 'wb') as f: # TODO: TAB THIS BY 1
                pickle.dump(training_data, f)
            print(f"\n  Saved to {output_file}")

    total_time = time.time() - t_start_all
    
    print("\n" + "="*60)
    print(f"  JOB {args.job_id} COMPLETE")
    print("="*60)
    print(f"  Successful samples: {successful} / {total}")
    print(f"  Success rate: {100*successful/total:.1f}%")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Avg time per sample: {total_time/max(successful,1):.1f}s")
    print("="*60)
    
    # Quick stats
    if successful > 0:
        eigs = [d['eig_opt'] for d in training_data]
        conv = sum(d['converged'] for d in training_data)
        print(f"\n  EIG range: {min(eigs):.2f} to {max(eigs):.2f}")
        print(f"  EIG mean: {np.mean(eigs):.2f} ± {np.std(eigs):.2f}")
        print(f"  Converged: {conv}/{successful}")
    
    # Save

    
    return training_data




# if __name__ == "__main__":
# def main():
generate_training_data()