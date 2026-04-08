#!/usr/bin/env python3
"""
Data reduction and preprocessing for OED training data.
Performs POD reduction on full velocity fields and adds noise to create noisy observations.
"""

import numpy as np
import pickle
import torch
from scipy.linalg import svd
import dolfin as dl
import os

# Add paths if needed
import sys
sys.path.append('.')
sys.path.append('generate_data/')
from generate_data.config import *
from generate_data.fe_setup import setup_fe_spaces


def load_training_data(filename):
    """Load training data from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both dictionary and list formats
    if isinstance(data, dict):
        if 'samples' in data:
            samples = data['samples']
            buildings = data.get('buildings', [])
            print(f"Loaded {len(samples)} samples from 'samples' key")
        else:
            samples = []
            print("No 'samples' key found")
    elif isinstance(data, list):
        samples = data
        print(f"Loaded {len(samples)} samples from list")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    return samples
def compute_pod_basis(wind_fields, n_modes=20):
    """
    Compute POD basis from full wind fields.
    
    Parameters
    ----------
    wind_fields : numpy array
        Matrix of wind fields (n_samples × n_dofs)
    n_modes : int
        Number of POD modes to retain
        
    Returns
    -------
    dict
        Dictionary containing POD basis, singular values, etc.
    """
    # X is (n_samples × n_dofs)
    X = wind_fields
    
    # Center the data
    mean_field = np.mean(X, axis=0)
    X_centered = X - mean_field
    
    # Compute SVD
    # U: (n_samples × n_samples), s: (min(n_samples, n_dofs)), Vt: (n_dofs × n_dofs)
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Keep first n_modes
    # For projecting new fields, we need the basis in DOF space: Vt[:n_modes].T
    basis = Vt[:n_modes].T  # Shape: (n_dofs, n_modes)
    s = s[:n_modes]
    
    explained_variance = np.cumsum(s**2 / np.sum(s**2))
    
    return {
        'basis': basis,  # POD basis in DOF space (n_dofs × n_modes)
        's': s,          # Singular values
        'n_modes': n_modes,
        'mean_field': mean_field,  # Mean field for reconstruction
        'explained_variance': explained_variance,
        'U': U,          # Left singular vectors (for reference)
        'Vt': Vt         # Right singular vectors (for reference)
    }


def project_to_pod(wind_field, pod_data):
    """
    Project a wind field onto POD basis.
    
    Parameters
    ----------
    wind_field : numpy array
        Flattened velocity field (n_dofs,)
    pod_data : dict
        POD data from compute_pod_basis
        
    Returns
    -------
    numpy array
        POD coefficients (n_modes,)
    """
    # Subtract mean
    wind_centered = wind_field - pod_data['mean_field']
    # Project onto basis (basis is n_dofs × n_modes)
    coeffs = np.dot(wind_centered, pod_data['basis'])
    return coeffs


def reconstruct_from_pod(coeffs, pod_data):
    """
    Reconstruct wind field from POD coefficients.
    
    Parameters
    ----------
    coeffs : numpy array
        POD coefficients (n_modes,)
    pod_data : dict
        POD data from compute_pod_basis
        
    Returns
    -------
    numpy array
        Reconstructed velocity field (n_dofs,)
    """
    return pod_data['mean_field'] + np.dot(coeffs, pod_data['basis'].T)


def add_relative_noise(clean_data, noise_level=0.01):
    """
    Add relative noise to data.
    
    Parameters
    ----------
    clean_data : numpy array
        Clean data (can be 1D or 2D)
    noise_level : float
        Relative noise level (e.g., 0.01 = 1% noise)
        
    Returns
    -------
    tuple
        (noisy_data, noise_variances)
    """
    clean_tensor = torch.from_numpy(clean_data).float()
    
    if clean_tensor.dim() == 1:
        # Handle 1D data
        max_val = torch.max(torch.abs(clean_tensor))
        noise_std = noise_level * max_val if max_val > 0 else noise_level
        noise = torch.randn_like(clean_tensor) * noise_std
        noisy_tensor = clean_tensor + noise
        noise_variances = noise_std ** 2 * torch.ones_like(clean_tensor)
    else:
        # Handle 2D data (multiple samples)
        noise_std = noise_level * torch.max(torch.abs(clean_tensor), dim=1, keepdim=True)[0]
        noise_std = torch.where(noise_std < 1e-6, torch.ones_like(noise_std) * noise_level, noise_std)
        noise = torch.randn_like(clean_tensor) * noise_std
        noisy_tensor = clean_tensor + noise
        noise_variances = (noise_std ** 2).squeeze()
    
    return noisy_tensor.numpy(), noise_variances.numpy()


def extract_full_wind_fields(samples):
    """
    Extract full velocity fields from training samples using wind_dof_vector.
    
    Parameters
    ----------
    samples : list
        List of training samples, each containing 'wind_dof_vector'
        
    Returns
    -------
    tuple
        (wind_fields_matrix, n_dofs)
    """
    n_samples = len(samples)
    
    # Get DOF dimension from first sample
    wind_dof_vector = samples[0]['wind_dof_vector']
    n_dofs = len(wind_dof_vector)
    
    # Initialize matrix to store all wind fields
    wind_fields = np.zeros((n_samples, n_dofs))
    
    print("Extracting full wind fields from samples...")
    for i, sample in enumerate(samples):
        # Store wind_dof_vector directly
        wind_fields[i, :] = sample['wind_dof_vector']
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")
    
    return wind_fields, n_dofs


def process_training_data(filename, output_file='data/mq_data_reduced.npz', 
                          noise_level=0.01, n_pod_modes=20):
    """
    Main function to process training data and create reduced dataset.
    
    Parameters
    ----------
    filename : str
        Input pickle file name
    output_file : str
        Output npz file name
    noise_level : float
        Relative noise level for adding noise
    n_pod_modes : int
        Number of POD modes to retain
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load data
    print(f"Loading {filename}...")
    samples = load_training_data(filename)
    n_samples = len(samples)
    print(f"Loaded {n_samples} samples")
    
    # Extract full wind fields from wind_dof_vector
    wind_fields, n_dofs = extract_full_wind_fields(samples)
    print(f"Wind field DOF dimension: {n_dofs}")
    
    # Extract other data
    param_dim = len(samples[0]['m_opt'])  # 4K+2
    
    # Initialize arrays for other data
    m_params = np.zeros((n_samples, param_dim))      # MAP parameters (optimized path)
    x_positions = np.zeros((n_samples, 2))           # Drone starting positions
    eig_K0 = np.zeros(n_samples)                     # EIG K0 (no observations)
    eig_K1 = np.zeros(n_samples)                     # EIG K1 (1 sensor)
    eig_K2 = np.zeros(n_samples)                     # EIG K2 (2 sensors)
    eig_K3 = np.zeros(n_samples)                     # EIG K3 (3 sensors)
    eig_gain = np.zeros(n_samples)                   # EIG gain (K3 - K0)
    speed_left = np.zeros(n_samples)                 # Left wall wind speed
    speed_right = np.zeros(n_samples)                # Right wall wind speed
    
    # Also collect initial guess parameters
    m_init_list = []
    
    print("Extracting remaining data...")
    for i, sample in enumerate(samples):
        # MAP parameters (Fourier coefficients for optimized path)
        m_params[i] = sample['m_opt']
        
        # Drone starting position
        x_positions[i] = sample['c_init']
        
        # EIG values
        eig_K0[i] = sample.get('eig_K0', sample.get('eig_init', 0.0))
        eig_K1[i] = sample.get('eig_K1', 0.0)
        eig_K2[i] = sample.get('eig_K2', 0.0)
        eig_K3[i] = sample.get('eig_K3', sample.get('eig_opt', 0.0))
        eig_gain[i] = eig_K3[i] - eig_K0[i]
        
        # Wind speeds at walls (from config or sample)
        speed_left[i] = sample.get('speed_left', 0.0)
        speed_right[i] = sample.get('speed_right', 0.0)
        
        # Store initial guess (if available, otherwise create placeholder)
        if 'm_init' in sample:
            m_init_list.append(sample['m_init'])
        else:
            m_init_list.append(np.zeros(param_dim))
    
    # Convert m_init to array
    m_init = np.array(m_init_list)
    
    # Compute POD basis from full wind fields
    print(f"\nComputing POD basis from full wind fields (n_modes={n_pod_modes})...")
    pod_data = compute_pod_basis(wind_fields, n_modes=n_pod_modes)
    
    # Calculate explained variance
    print(f"POD explained variance for first {n_pod_modes} modes:")
    for i in range(min(10, n_pod_modes)):
        print(f"  Mode {i+1}: {pod_data['explained_variance'][i]:.4f}")
    print(f"  Total explained: {pod_data['explained_variance'][-1]:.4f}")
    
    # Project all wind fields to POD space
    print("Projecting wind fields to POD space...")
    v_reduced = np.zeros((n_samples, n_pod_modes))
    for i in range(n_samples):
        v_reduced[i] = project_to_pod(wind_fields[i], pod_data)
        if (i + 1) % 100 == 0:
            print(f"  Projected {i+1}/{n_samples} samples")
    
    # Add noise to create noisy observations
    print(f"\nAdding noise (level={noise_level})...")
    
    # 1. Noisy drone positions
    x_data, x_noise_var = add_relative_noise(x_positions, noise_level)
    
    # 2. Noisy reduced velocity (POD coefficients)
    v_data, v_noise_var = add_relative_noise(v_reduced, noise_level)
    
    # For v_mean, we don't have explicit means from wind_dof_vector, so we'll compute
    # approximate mean velocities from the wind fields
    # For now, use zeros or compute from first few DOFs (mean vx, vy)
    # This is a placeholder - you may want to compute actual means from the field
    v_mean = np.zeros((n_samples, 2))
    v_mean_data = np.zeros((n_samples, 2))
    
    # Simple approach: take average of first few DOFs as rough estimate
    # Each DOF contains both x and y components in vector space
    # For VectorFunctionSpace, DOFs alternate: vx0, vy0, vx1, vy1, ...
    for i in range(n_samples):
        # Rough estimate: average of all x-components and all y-components
        vx_components = wind_fields[i, 0::2]  # Every other DOF starting from 0
        vy_components = wind_fields[i, 1::2]  # Every other DOF starting from 1
        v_mean[i, 0] = np.mean(vx_components)
        v_mean[i, 1] = np.mean(vy_components)
    
    # Add noise to mean velocities
    v_mean_data, v_mean_noise_var = add_relative_noise(v_mean, noise_level)
    
    # Save all data to npz file
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        # Clean/true data
        m=m_params,                    # MAP parameters (optimized path coefficients)
        m_init=m_init,                 # Initial path coefficients
        x=x_positions,                 # True drone starting positions
        v=v_reduced,                   # True reduced velocity (POD coefficients)
        v_mean=v_mean,                 # True mean wind velocities (approx)
        wind_dofs=wind_fields,         # Full wind field DOFs (for reference)
        
        # EIG values
        eig_K0=eig_K0,                 # EIG with no observations
        eig_K1=eig_K1,                 # EIG with 1 sensor
        eig_K2=eig_K2,                 # EIG with 2 sensors
        eig_K3=eig_K3,                 # EIG with 3 sensors
        eig_gain=eig_gain,             # EIG gain (K3 - K0)
        
        # Wind parameters
        speed_left=speed_left,         # Wind speed at left wall
        speed_right=speed_right,       # Wind speed at right wall
        
        # Noisy data (for training)
        x_data=x_data,                 # Noisy drone starting positions
        v_data=v_data,                 # Noisy reduced velocity (POD coefficients)
        v_mean_data=v_mean_data,       # Noisy mean wind velocities
        
        # Noise variances
        x_noise_var=x_noise_var,
        v_noise_var=v_noise_var,
        v_mean_noise_var=v_mean_noise_var,
        
        # POD metadata
        pod_basis=pod_data['basis'],
        pod_singular_values=pod_data['s'],
        pod_mean_field=pod_data['mean_field'],
        pod_explained_variance=pod_data['explained_variance'],
        n_pod_modes=n_pod_modes,
        
        # Other metadata
        param_dim=param_dim,
        n_samples=n_samples,
        noise_level=noise_level,
        n_dofs=n_dofs
    )
    
    print("\n" + "="*60)
    print("DATA REDUCTION COMPLETE")
    print("="*60)
    print(f"  Output file: {output_file}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Parameter dimension: {param_dim}")
    print(f"  POD modes: {n_pod_modes}")
    print(f"  Wind field DOFs: {n_dofs}")
    print(f"\n  Keys in file:")
    with np.load(output_file) as data_loaded:
        for key in data_loaded.keys():
            if key not in ['pod_basis', 'pod_mean_field', 'wind_dofs']:  # Skip large arrays
                print(f"    - {key}: {data_loaded[key].shape}")
    print("="*60)
    
    return output_file


def load_reduced_data(filename='data/mq_data_reduced.npz'):
    """
    Load reduced data from npz file.
    
    Returns
    -------
    dict
        Dictionary with all data
    """
    data = np.load(filename, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def test_reconstruction(data_file='data/mq_data_reduced.npz', n_samples=5):
    """
    Test reconstruction from reduced representation.
    """
    data = load_reduced_data(data_file)
    
    print("\n" + "="*60)
    print("RECONSTRUCTION TEST")
    print("="*60)
    
    # Reconstruct pod_data dict
    pod_data = {
        'basis': data['pod_basis'],
        'mean_field': data['pod_mean_field'],
        'n_modes': data['n_pod_modes'],
        's': data['pod_singular_values']
    }
    
    # Test reconstruction for a few samples
    errors = []
    for i in range(min(n_samples, data['n_samples'])):
        # Reconstruct full wind field from POD coefficients
        v_reconstructed = reconstruct_from_pod(data['v'][i], pod_data)
        
        # Compare with original wind field (if we have it)
        if 'wind_dofs' in data:
            original = data['wind_dofs'][i]
            error = np.linalg.norm(v_reconstructed - original) / np.linalg.norm(original)
            errors.append(error)
            print(f"Sample {i}: Reconstruction relative error = {error:.6f}")
        else:
            print(f"Sample {i}: Successfully reconstructed from {data['n_pod_modes']} POD modes")
            errors.append(0.0)
    
    if errors and errors[0] > 0:
        print(f"\nAverage reconstruction error: {np.mean(errors):.6f}")
    
    return errors

def plot_reconstruction_error(data_file='data/mq_data_reduced.npz', max_modes=None, n_test_samples=10):
    """
    Plot reconstruction error as a function of number of POD modes.
    
    Parameters
    ----------
    data_file : str
        Path to the reduced data file
    max_modes : int
        Maximum number of modes to test (default: min(100, n_pod_modes))
    n_test_samples : int
        Number of test samples to use for error calculation
    """
    import matplotlib.pyplot as plt
    
    data = load_reduced_data(data_file)
    
    # Get maximum modes to test
    n_total_modes = data['n_pod_modes']
    if max_modes is None:
        max_modes = min(100, n_total_modes)
    else:
        max_modes = min(max_modes, n_total_modes)
    
    # Get test samples
    n_samples = min(n_test_samples, data['n_samples'])
    indices = np.random.choice(data['n_samples'], n_samples, replace=False)
    
    # Reconstruct pod_data dict
    pod_data = {
        'basis': data['pod_basis'],
        'mean_field': data['pod_mean_field'],
        'n_modes': n_total_modes,
        's': data['pod_singular_values']
    }
    
    # Calculate error for different numbers of modes
    n_modes_list = list(range(1, max_modes + 1))
    mean_errors = []
    std_errors = []
    
    print(f"\nComputing reconstruction errors for {max_modes} mode counts...")
    for n_modes in n_modes_list:
        # Use only first n_modes
        pod_data_subset = {
            'basis': pod_data['basis'][:, :n_modes],
            'mean_field': pod_data['mean_field'],
            'n_modes': n_modes,
            's': pod_data['s'][:n_modes]
        }
        
        errors = []
        for idx in indices:
            # Reconstruct full wind field from POD coefficients
            v_reconstructed = reconstruct_from_pod(data['v'][idx, :n_modes], pod_data_subset)
            
            # Compare with original wind field
            original = data['wind_dofs'][idx]
            error = np.linalg.norm(v_reconstructed - original) / np.linalg.norm(original)
            errors.append(error)
        
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))
        
        if n_modes % 20 == 0:
            print(f"  Modes: {n_modes}/{max_modes}, Mean error: {mean_errors[-1]:.6f}")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reconstruction error vs number of modes
    ax1 = axes[0]
    ax1.plot(n_modes_list, mean_errors, 'b-', linewidth=2, label='Mean error')
    ax1.set_yscale('log')
    ax1.fill_between(n_modes_list, 
                     np.array(mean_errors) - np.array(std_errors),
                     np.array(mean_errors) + np.array(std_errors),
                     alpha=0.3, color='blue', label='±1 std')
    ax1.set_xlabel('Number of POD Modes', fontsize=12)
    ax1.set_ylabel('Relative Reconstruction Error', fontsize=12)
    ax1.set_title('Reconstruction Error vs POD Modes', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add horizontal line at 1% error
    ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% error')
    
    # Plot 2: Explained variance vs number of modes
    ax2 = axes[1]
    explained_variance = data['pod_explained_variance'][:max_modes]
    ax2.plot(n_modes_list, explained_variance, 'g-', linewidth=2, label='Cumulative explained variance')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of POD Modes', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Explained Variance vs POD Modes', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add horizontal line at 95% and 99%
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='99%')
    
    plt.tight_layout()
    plt.savefig('pod_reconstruction_error.png', dpi=150)
    print(f"\nPlot saved to pod_reconstruction_error.png")
    plt.show()
    
    # Print optimal mode selection
    print("\n" + "="*60)
    print("OPTIMAL MODE SELECTION")
    print("="*60)
    
    # Find modes needed for different error thresholds
    for target_error in [0.1, 0.05, 0.01, 0.005]:
        for i, err in enumerate(mean_errors):
            if err < target_error:
                print(f"  To achieve {target_error*100:.1f}% error: {i+1} modes needed")
                break
        else:
            print(f"  To achieve {target_error*100:.1f}% error: >{max_modes} modes needed")
    
    print("\n  For 95% variance explained:")
    for i, var in enumerate(explained_variance):
        if var >= 0.95:
            print(f"    {i+1} modes explain {var*100:.1f}% of variance")
            break
    
    print("\n  For 99% variance explained:")
    for i, var in enumerate(explained_variance):
        if var >= 0.99:
            print(f"    {i+1} modes explain {var*100:.1f}% of variance")
            break
    

if __name__ == "__main__":
    
    # Use your actual data file
    filename = 'generate_data/oed_training_hippylib.pkl'  # Adjust path as needed
    output_file = 'data/mq_data_reduced.npz'
    
    # Process data with POD reduction
    # noise_level=0.0 for clean data, >0 for noisy observations
    process_training_data(filename, output_file, noise_level=0.00, n_pod_modes=200)
    
    # Test reconstruction
    test_reconstruction(output_file)
    
    # Plot reconstruction error vs number of modes
    plot_reconstruction_error(output_file, max_modes=200, n_test_samples=20)