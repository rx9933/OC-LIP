"""Training data generation for OED with varying wind and drone positions."""

import numpy as np
import dolfin as dl
import pickle
import time
from scipy.optimize import minimize

from config import *
from fourier_utils import fourier_frequencies, xbar_coeffs_to_m
from wind_utils import sample_spectral_wind, coeffs_to_nn_input, nn_input_dim
from fe_utils import reset_cached_bbt
from oed_objective import CachedEigensolver, oed_objective_and_grad
from fe_setup import setup_fe_spaces, setup_prior

# Suppress FEniCS logging
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


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


def create_initial_guess(c_init, K, amp=0.05):
    """
    Create initial Fourier parameter guess centered at drone position.
    
    Parameters
    ----------
    c_init : np.ndarray
        (2,) initial drone position
    K : int
        Number of Fourier modes
    amp : float
        Initial amplitude for Fourier coefficients
        
    Returns
    -------
    np.ndarray
        Flat Fourier parameter vector
    """
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]
    m0[1] = c_init[1]
    m0[2] = amp      # θ₁
    m0[5] = amp      # η₁
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
        _, _, eig_init, _, _, _ = oed_objective_and_grad(
            m0, Vh, mesh, prior, simulation_times, observation_times,
            wind_velocity, K, omegas, r_modes, noise_variance, t_param,
            eigsolver, obstacles=None, include_penalties=True 
        )
        print(f"    Initial EIG = {eig_init:.2f}")
        # ===== END OF ADDED SECTION =====
        
        # Define objective function for this sample
        def objective(m):
            J, grad, eig_val, pen_val, spd_val, elapsed = oed_objective_and_grad(
                m, Vh, mesh, prior, simulation_times, observation_times,
                wind_velocity, K, omegas, r_modes, noise_variance, t_param,
                eigsolver, obstacles=None, include_penalties=False
            )
            return J, grad
        
        # Optimize
        result = minimize(
            objective, m0,
            jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': OPT_MAXITER, 'disp': False,
                     'ftol': OPT_FTOL, 'maxls': OPT_MAXLS}
        )
        
        m_opt = result.x
        
        # Compute final EIG (without penalties)
        eigsolver.reset()
        _, _, eig_opt, _, _, _ = oed_objective_and_grad(
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
        
        return sample, True
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED: {e}")
        return None, False


def generate_training_data(n_samples=N_SAMPLES, output_file=OUTPUT_FILE):
    """
    Generate training data for OED.
    
    Parameters
    ----------
    n_samples : int
        Number of wind samples to generate
    output_file : str
        Path to save output pickle file
        
    Returns
    -------
    list
        List of training samples
    """
    print("="*60)
    print("  OED TRAINING DATA GENERATION")
    print("="*60)
    print(f"  Number of samples: {n_samples}")
    print(f"  Mean_vx distribution: N({MEAN_VX_MEAN}, {MEAN_VX_STD}²)")
    print(f"  Drone position distribution: N({DRONE_POS_MEAN}, {DRONE_POS_STD}²)")
    print("="*60)
    
    base_seed = int(time.time())
    print(f"Using base seed: {base_seed} (based on current time)")
    max_attempts = n_samples * 3
    
    # Setup
    mesh, Vh, wind_velocity = setup_fe_spaces()
    prior = setup_prior(Vh)
    
    # Compute frequencies
    omegas = fourier_frequencies(TY, K)
    
    print(f"\n  Mesh DOFs: {Vh.dim()}")
    print(f"  Fourier modes: {K} (parameter dim: {4*K + 2})")
    print(f"  NN input dim: {nn_input_dim(WIND_R)}")
    
    training_data = []
    total = n_samples
    count = 0
    successful = 0
    t_start_all = time.time()
    
    # for seed in range(n_samples):
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
        
        sample, success = generate_training_sample(
            seed, mesh, Vh, prior, SIMULATION_TIMES, OBSERVATION_TIMES,
            OBSERVATION_TIMES, K, omegas, R_MODES, NOISE_VARIANCE,
            BOUNDS, wind_params
        )
        
        # Find this section in generate_training_data function:
        if success:
            training_data.append(sample)
            successful += 1
            # ===== MODIFY THIS PRINT STATEMENT =====
            print(f"    → EIG: {sample['eig_init']:.2f} → {sample['eig_opt']:.2f} "
                f"(gain={sample['eig_gain']:.2f}) conv={sample['converged']} "
                f"[{sample['time']:.1f}s]")
        # ===== END MODIFICATION =====
        # Save data
    total_time = time.time() - t_start_all
    
    print("\n" + "="*60)
    print("  TRAINING DATA GENERATION COMPLETE")
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
    with open(output_file, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"\n  Saved to {output_file}")
    
    return training_data


if __name__ == "__main__":
    generate_training_data()