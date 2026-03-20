"""Moving sensor misfit class for time-dependent observations."""

import dolfin as dl
import numpy as np
import sys
sys.path.append('../../../')
from hippylib import assemblePointwiseObservation, STATE, TimeDependentVector


class MovingSensorMisfit:
    """
    Moving sensor: at observation time t_k, only target k is active.
    Matches SpaceTimePointwiseStateObservation interface exactly.
    """
    
    def __init__(self, Vh, observation_times, targets):
        """
        Initialize moving sensor misfit.
        
        Parameters
        ----------
        Vh : dolfin FunctionSpace
            Finite element space
        observation_times : np.ndarray
            Times at which observations are taken
        targets : np.ndarray
            (n_obs, 2) array of sensor positions
        """
        self.Vh = Vh
        self.observation_times = observation_times
        self.n_obs = len(observation_times)
        self.targets = targets
        self.noise_variance = 1.0
        
        assert targets.shape[0] == self.n_obs
        
        # One B_k (1 × N_dof) per observation time
        self.B_list = []
        for k in range(self.n_obs):
            B_k = assemblePointwiseObservation(Vh, targets[k:k+1, :])
            self.B_list.append(B_k)
        
        # Also store B_list[0] as self.B so hIPPYlib internals don't break
        self.B = self.B_list[0]
        
        # Data: obs-space vectors (dim 1) at each time
        self.d = TimeDependentVector(observation_times)
        self.d.data = []
        for k in range(self.n_obs):
            v = dl.Vector()
            self.B.init_vector(v, 0)
            self.d.data.append(v)
        
        # Pre-allocate work vectors
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)   # N_dof
        self.B.init_vector(self.Bu_snapshot, 0)  # dim 1
        self.B.init_vector(self.d_snapshot, 0)   # dim 1
    
    def observe(self, x, d):
        """Observe state at sensor positions."""
        for k in range(self.n_obs):
            tk = self.observation_times[k]
            x[STATE].retrieve(self.u_snapshot, tk)
            self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
            d.store(self.Bu_snapshot, tk)
    
    def cost(self, x):
        """Compute misfit cost."""
        c = 0.0
        for k in range(self.n_obs):
            tk = self.observation_times[k]
            x[STATE].retrieve(self.u_snapshot, tk)
            self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, tk)
            self.Bu_snapshot.axpy(-1.0, self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
        return 0.5 * c / self.noise_variance
    
    def grad(self, i, x, out):
        """Compute gradient with respect to state."""
        out.zero()
        if i == STATE:
            for k in range(self.n_obs):
                tk = self.observation_times[k]
                x[STATE].retrieve(self.u_snapshot, tk)
                self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, tk)
                self.Bu_snapshot.axpy(-1.0, self.d_snapshot)
                self.Bu_snapshot *= 1.0 / self.noise_variance
                self.B_list[k].transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, tk)
        else:
            pass
    
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """Set linearization point (required by hIPPYlib)."""
        pass
    
    def apply_ij(self, i, j, direction, out):
        """Apply Hessian-vector product."""
        out.zero()
        if i == STATE and j == STATE:
            for k in range(self.n_obs):
                tk = self.observation_times[k]
                direction.retrieve(self.u_snapshot, tk)
                self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1.0 / self.noise_variance
                self.B_list[k].transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, tk)
        else:
            pass