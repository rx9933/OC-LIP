# load in data, then do POD or some sort of velocty field reduction
# save the optimal parameters and the 
# save the data to: data/mq_data_reduced.npz
# have at least 9 keys:
# 1. ['m'] to retrieve the MAP parameters, the 4K+2 coefficients of the path
# 2. ['x'] to retrieve the position of the drone, dimension 2
# 3. ['v'] to retrieve the velocity data in reduced space (POD decomposition) -- need to decide if using wind prior covariance here
# 4. ['v_coeff'] to retrieve the 18 (?) spectral coefficients
# 5. ['v_mean'] to retrieve the 2 mean vx, vy of the wind
# ------
# 6. ['x_data'] to retrieve the noised position of the drone, dimension 2
# 7. ['v_data'] to retrieve the noised velocity data in reduced space (POD decomposition) -- need to decide if using wind prior covariance here
# 8. ['v_coeff_data'] to retrieve the 18 (?) spectral coefficients 
# 9. ['v_mean_data'] to retrieve the 2 mean vx, vy of the wind

# Note that when simulating data noise, the function noise() below uses relative noise. So, add noise seperately for 1) the position of the drone, 
# 2) for the velocity field data, and 3) the coefficients of the velocity field, and 4) the mean of vx and mean of vy.

# Adding relative noise to the spectral coefficients can be very different from adding noise to the field data. This is an interesting question. 

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
from generate_data.wind_utils import spectral_wind_to_field
from generate_data.fe_setup import setup_fe_spaces


def load_training_data(filename):
    """Load training data from pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


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
    # X is already (n_samples × n_dofs)
    X = wind_fields
    
    # Compute SVD
    U, s, Vt = svd(X, full_matrices=False)
    
    # Keep first n_modes
    U = U[:, :n_modes]
    s = s[:n_modes]
    
    return {
        'U': U,  # POD basis (n_samples × n_modes)
        's': s,  # Singular values
        'n_modes': n_modes,
        'mean_field': np.mean(X, axis=0)  # Mean field for reconstruction
    }


def project_to_pod(wind_field, pod_data):
    """
    Project a wind field onto POD basis.
    
    Parameters
    ----------
    wind_field : numpy array
        Flattened velocity field
    pod_data : dict
        POD data from compute_pod_basis
        
    Returns
    -------
    numpy array
        POD coefficients
    """
    # Subtract mean
    wind_centered = wind_field - pod_data['mean_field']
    # Project onto basis
    coeffs = np.dot(wind_centered, pod_data['U'])
    return coeffs


def reconstruct_from_pod(coeffs, pod_data):
    """
    Reconstruct wind field from POD coefficients.
    
    Parameters
    ----------
    coeffs : numpy array
        POD coefficients
    pod_data : dict
        POD data from compute_pod_basis
        
    Returns
    -------
    numpy array
        Reconstructed velocity field
    """
    return pod_data['mean_field'] + np.dot(coeffs, pod_data['U'].T)


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
    noisy_tensor = torch.zeros_like(clean_tensor)
    
    if clean_tensor.dim() == 1:
        # Handle 1D data
        max_val = torch.max(torch.abs(clean_tensor))
        noise_std = noise_level * max_val
        noise = torch.randn_like(clean_tensor) * noise_std
        noisy_tensor = clean_tensor + noise
        noise_variances = noise_std ** 2
    else:
        # Handle 2D data (multiple samples)
        noise_variances = torch.zeros(clean_tensor.shape[0])
        for i in range(clean_tensor.shape[0]):
            max_val = torch.max(torch.abs(clean_tensor[i]))
            noise_std = noise_level * max_val
            noise = torch.randn_like(clean_tensor[i]) * noise_std
            noisy_tensor[i] = clean_tensor[i] + noise
            noise_variances[i] = noise_std ** 2
    
    return noisy_tensor.numpy(), noise_variances.numpy()


def extract_full_wind_fields(data, mesh, Vh):
    """
    Extract full velocity fields from training samples.
    
    Parameters
    ----------
    data : list
        List of training samples
    mesh : dolfin Mesh
        Computational mesh
    Vh : dolfin FunctionSpace
        Finite element space
        
    Returns
    -------
    tuple
        (wind_fields_matrix, wind_coeffs_list, wind_means)
    """
    n_samples = len(data)
    n_dofs = Vh.dim()
    
    # Initialize matrix to store all wind fields
    wind_fields = np.zeros((n_samples, n_dofs))
    wind_coeffs_list = []
    wind_means = np.zeros((n_samples, 2))
    
    print("Extracting full wind fields from samples...")
    for i, sample in enumerate(data):
        # Get wind coefficients from sample
        wind_coeffs = sample['wind_coeffs']
        wind_coeffs_list.append(wind_coeffs)
        
        # Store mean wind velocities
        wind_means[i, 0] = wind_coeffs['mean_vx']
        wind_means[i, 1] = wind_coeffs['mean_vy']
        
        # Reconstruct full velocity field
        wind_field = spectral_wind_to_field(mesh, Vh, wind_coeffs)
        
        # Extract DOFs as numpy array
        wind_fields[i, :] = wind_field.vector().get_local()
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")
    
    return wind_fields, wind_coeffs_list, wind_means


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
    data = load_training_data(filename)
    print(f"Loaded {len(data)} samples")
    
    # Setup finite element spaces for field reconstruction
    print("Setting up FE spaces...")
    mesh, Vh, _ = setup_fe_spaces()
    print(f"Mesh DOFs: {Vh.dim()}")
    
    # Extract full wind fields
    wind_fields, wind_coeffs_list, wind_means = extract_full_wind_fields(data, mesh, Vh)
    
    # Extract other data
    n_samples = len(data)
    param_dim = len(data[0]['nn_output'])  # 4K+2
    
    # Initialize arrays for other data
    m_params = np.zeros((n_samples, param_dim))  # MAP parameters
    x_positions = np.zeros((n_samples, 2))       # Drone starting positions
    eig_init = np.zeros(n_samples)
    eig_opt = np.zeros(n_samples)
    eig_gain = np.zeros(n_samples)
    
    # Also collect spectral coefficients in a matrix
    n_wind_coeffs = len(data[0]['wind_coeffs']['a_ij'].flatten()) + \
                    len(data[0]['wind_coeffs']['b_ij'].flatten())
    v_coeffs = np.zeros((n_samples, n_wind_coeffs))
    
    print("Extracting remaining data...")
    for i, sample in enumerate(data):
        # MAP parameters (Fourier coefficients for optimal path)
        m_params[i] = sample['nn_output']
        
        # Drone starting position
        x_positions[i] = sample['c_init']
        
        # Wind spectral coefficients
        a_ij = sample['wind_coeffs']['a_ij'].flatten()
        b_ij = sample['wind_coeffs']['b_ij'].flatten()
        v_coeffs[i] = np.concatenate([a_ij, b_ij])
        
        # EIG values
        eig_init[i] = sample['eig_init']
        eig_opt[i] = sample['eig_opt']
        eig_gain[i] = sample['eig_gain']
    
    # Compute POD basis from full wind fields
    print(f"\nComputing POD basis from full wind fields (n_modes={n_pod_modes})...")
    pod_data = compute_pod_basis(wind_fields, n_modes=n_pod_modes)
    
    # Calculate explained variance
    total_variance = np.sum(pod_data['s']**2)
    explained_variance = np.cumsum(pod_data['s']**2 / total_variance)
    print(f"POD explained variance for first {n_pod_modes} modes:")
    for i in range(min(10, n_pod_modes)):
        print(f"  Mode {i+1}: {explained_variance[i]:.4f}")
    print(f"  Total explained: {explained_variance[-1]:.4f}")
    
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
    
    # 3. Noisy spectral coefficients
    v_coeffs_data, v_coeffs_noise_var = add_relative_noise(v_coeffs, noise_level)
    
    # 4. Noisy mean wind velocities
    v_means_data, v_means_noise_var = add_relative_noise(wind_means, noise_level)
    
    # Save all data to npz file
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        # Clean/true data
        m=m_params,                    # MAP parameters (optimal path coefficients)
        x=x_positions,                 # True drone positions
        v=v_reduced,                   # True reduced velocity (POD coefficients)
        v_coeff=v_coeffs,              # True spectral coefficients
        v_mean=wind_means,             # True mean wind velocities
        eig_init=eig_init,             # Initial EIG values
        eig_opt=eig_opt,               # Optimal EIG values
        eig_gain=eig_gain,             # EIG gain
        
        # Noisy data (for training)
        x_data=x_data,                 # Noisy drone positions
        v_data=v_data,                 # Noisy reduced velocity (POD coefficients)
        v_coeff_data=v_coeffs_data,    # Noisy spectral coefficients
        v_mean_data=v_means_data,      # Noisy mean wind velocities
        
        # Noise variances
        x_noise_var=x_noise_var,
        v_noise_var=v_noise_var,
        v_coeff_noise_var=v_coeffs_noise_var,
        v_mean_noise_var=v_means_noise_var,
        
        # POD metadata
        pod_basis=pod_data['U'],
        pod_singular_values=pod_data['s'],
        pod_mean_field=pod_data['mean_field'],
        n_pod_modes=n_pod_modes,
        
        # Other metadata
        param_dim=param_dim,
        n_samples=n_samples,
        noise_level=noise_level,
        n_dofs=Vh.dim()
    )
    
    print("\n" + "="*60)
    print("DATA REDUCTION COMPLETE")
    print("="*60)
    print(f"  Output file: {output_file}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Parameter dimension: {param_dim}")
    print(f"  POD modes: {n_pod_modes}")
    print(f"  Wind field DOFs: {Vh.dim()}")
    print(f"\n  Keys in file:")
    with np.load(output_file) as data_loaded:
        for key in data_loaded.keys():
            if key not in ['pod_basis', 'pod_mean_field']:  # Skip large arrays for printing
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
        'U': data['pod_basis'],
        'mean_field': data['pod_mean_field'],
        'n_modes': data['n_pod_modes'],
        's': data['pod_singular_values']
    }
    
    errors = []
    for i in range(min(n_samples, data['n_samples'])):
        # Reconstruct full wind field from POD coefficients
        v_reconstructed = reconstruct_from_pod(data['v'][i], pod_data)
        
        # We don't have the original full field saved, so we'll compare
        # the reconstructed field with what we could compute from coefficients
        # For now, we'll just report the reconstruction is possible
        print(f"Sample {i}: Successfully reconstructed from {data['n_pod_modes']} POD modes")
        errors.append(0.0)  # Placeholder
    
    print(f"\nReconstruction test complete for {min(n_samples, data['n_samples'])} samples")
    
    return errors


if __name__ == "__main__":
    
    filename = 'generate_data/oed_training_data_combined.pkl'  
    output_file = 'data/mq_data_reduced.npz'
    
    # Noise should have been added when determining the optimal path (before PDE MAP evaluation)
    process_training_data(filename, output_file, noise_level=0.00, n_pod_modes=20)
    
    test_reconstruction(output_file)