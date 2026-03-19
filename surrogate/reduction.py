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


import torch
def noise(clean_data, noise_level=0.01): # change noise_level
    noisy_data = torch.zeros_like(clean_data)
    noise_variances = torch.zeros(clean_data.shape[0])
    
    for i in range(clean_data.shape[0]):
        # Get maximum absolute value (L∞ norm)
        max_val = torch.max(torch.abs(clean_data[i]))
        
        # Compute noise standard deviation
        noise_std = noise_level * max_val
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_data[i]) * noise_std
        noisy_data[i] = clean_data[i] + noise
        
        # Store noise variance
        noise_variances[i] = noise_std ** 2
    return noisy_data