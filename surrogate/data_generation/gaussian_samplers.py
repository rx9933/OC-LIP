import numpy as np

def sample_mean_vx(mu=0.5, sigma=0.2):
    """Sample mean horizontal wind from Gaussian."""
    return np.random.normal(mu, sigma)


def sample_drone_position(mu=np.array([0.5, 0.5]), sigma=0.2):
    """
    Sample drone initial position from Gaussian, clipped to [0,1]^2.
    """
    c = np.random.normal(mu, sigma, size=2)
    return np.clip(c, 0.05, 0.95)   # avoid boundary issues