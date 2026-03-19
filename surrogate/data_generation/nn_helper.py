import numpy as np

def coeffs_to_nn_input(coeffs, c_init):
    """
    Flatten spectral coefficients + drone position into a single NN input vector.
    
    Layout: [mean_vx, mean_vy, a_11, a_12, ..., a_rr, b_11, b_12, ..., b_rr, cx, cy]
    
    Size: 2 + 2*r_wind^2 + 2
    """
    a_flat = coeffs['a_ij'].flatten()
    b_flat = coeffs['b_ij'].flatten()
    return np.concatenate([
        [coeffs['mean_vx'], coeffs['mean_vy']],
        a_flat,
        b_flat,
        c_init
    ])


def nn_input_dim(r_wind):
    """Dimension of the NN input vector."""
    return 2 + 2 * r_wind**2 + 2