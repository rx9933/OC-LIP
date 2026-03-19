"""Core OED functions: Fourier parameterization, penalties, misfit."""

import numpy as np
import dolfin as dl
import ufl
"""Core OED functions: Fourier parameterization, penalties, misfit."""
import sys
sys.path.append('../../../')
# Import from hippylib
from hippylib.hippylib.modeling.variables import (STATE, PARAMETER, ADJOINT)
sys.path.append('../../')
from hippylib.hippylib import assemblePointwiseObservation, TimeDependentVector
from oed_solver import *
# Global cache for bounding box tree
_cached_bbt = None

# ═══════════════════════════════════════════════════════════════
# 8.  BUILD PROBLEM  
# ═══════════════════════════════════════════════════════════════
class MovingSensorMisfit:
    """
    Moving sensor: at observation time t_k, only target k is active.
    Matches SpaceTimePointwiseStateObservation interface exactly.
    """
    
    def __init__(self, Vh, observation_times, targets):
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
        self.u_snapshot  = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)   # N_dof
        self.B.init_vector(self.Bu_snapshot, 0)  # dim 1
        self.B.init_vector(self.d_snapshot, 0)   # dim 1
    
    def observe(self, x, d):
        for k in range(self.n_obs):
            tk = self.observation_times[k]
            x[STATE].retrieve(self.u_snapshot, tk)
            self.B_list[k].mult(self.u_snapshot, self.Bu_snapshot)
            d.store(self.Bu_snapshot, tk)
    
    def cost(self, x):
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
        pass
    
    def apply_ij(self, i, j, direction, out):
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


def build_problem(m_fourier):
    targets = generate_targets(m_fourier, t_param, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets)
    misfit.noise_variance = noise_variance

    problem = TimeDependentAD(
        mesh, [Vh, Vh, Vh], prior, misfit,
        simulation_times, wind_velocity, True
    )
    return problem, misfit, targets

# ============================================================================
# FOURIER PATH PARAMETERIZATION
# ============================================================================

def fourier_path(t, xbar, coeffs, omegas):
    """Evaluate Fourier curve at parameter t."""
    x = xbar[0]
    y = xbar[1]
    for k, w in enumerate(omegas):
        x += coeffs[k, 0] * np.cos(w * t) + coeffs[k, 1] * np.sin(w * t)
        y += coeffs[k, 2] * np.cos(w * t) + coeffs[k, 3] * np.sin(w * t)
    return np.array([x, y])


def m_to_xbar_coeffs(m_fourier, K):
    """Unpack flat optimisation vector → (xbar, coeffs)."""
    xbar = m_fourier[:2].copy()
    coeffs = np.zeros((K, 4))
    for k in range(K):
        coeffs[k, 0] = m_fourier[2 + 4*k]      # θ_k
        coeffs[k, 1] = m_fourier[3 + 4*k]      # φ_k
        coeffs[k, 2] = m_fourier[4 + 4*k]      # ψ_k
        coeffs[k, 3] = m_fourier[5 + 4*k]      # η_k
    return xbar, coeffs


def xbar_coeffs_to_m(xbar, coeffs, K):
    """Pack (xbar, coeffs) → flat optimisation vector."""
    m = np.zeros(4*K + 2)
    m[:2] = xbar
    for k in range(K):
        m[2 + 4*k] = coeffs[k, 0]
        m[3 + 4*k] = coeffs[k, 1]
        m[4 + 4*k] = coeffs[k, 2]
        m[5 + 4*k] = coeffs[k, 3]
    return m


def generate_targets(m_fourier, t_param, K, omegas):
    """Generate target positions at given times."""
    xbar, coeffs = m_to_xbar_coeffs(m_fourier, K)
    targets = np.array([fourier_path(t, xbar, coeffs, omegas)
                        for t in t_param])
    eps = 1e-6
    n_clipped = np.sum(targets < eps) + np.sum(targets > 1.0 - eps)
    if n_clipped > 0:
        print(f"  WARNING: path clipped at {n_clipped} coordinates")
    targets[:, 0] = np.clip(targets[:, 0], eps, 1.0 - eps)
    targets[:, 1] = np.clip(targets[:, 1], eps, 1.0 - eps)
    return targets


def fourier_velocity(m_fourier, t_arr, K, omegas):
    """Compute dx/dt, dy/dt at each time in t_arr."""
    xbar, coeffs = m_to_xbar_coeffs(m_fourier, K)
    vx = np.zeros(len(t_arr))
    vy = np.zeros(len(t_arr))
    for k, w in enumerate(omegas):
        vx += w * (-coeffs[k, 0] * np.sin(w * t_arr) + coeffs[k, 1] * np.cos(w * t_arr))
        vy += w * (-coeffs[k, 2] * np.sin(w * t_arr) + coeffs[k, 3] * np.cos(w * t_arr))
    return vx, vy


# ============================================================================
# PENALTIES
# ============================================================================

def boundary_penalty_dense(m_fourier, t_param, K, omegas, 
                           zeta=None, eps_bdy=0.02, n_dense=200):
    """Evaluate boundary penalty on a dense grid."""
    if zeta is None:
        zeta = 1000.0
        
    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]
    targets_dense = generate_targets(m_fourier, t_dense, K, omegas)
    
    val = 0.0
    S_B_dense = np.zeros((n_dense, 2))
    margin = 0.05
    for d in range(2):
        for j in range(n_dense):
            lo = targets_dense[j, d] - eps_bdy
            hi = (1.0 - eps_bdy) - targets_dense[j, d]
            if lo < margin:
                val += dt_dense * zeta * (margin - lo)**2
                S_B_dense[j, d] -= zeta * 2.0 * (margin - lo)
            if hi < margin:
                val += dt_dense * zeta * (margin - hi)**2
                S_B_dense[j, d] += zeta * 2.0 * (margin - hi)
    
    # Project onto Fourier
    g = np.zeros(4*K + 2)
    g[0] = dt_dense * np.sum(S_B_dense[:, 0])
    g[1] = dt_dense * np.sum(S_B_dense[:, 1])
    for kk in range(K):
        cos_v = np.cos(omegas[kk] * t_dense)
        sin_v = np.sin(omegas[kk] * t_dense)
        g[2 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 0], cos_v)
        g[3 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 0], sin_v)
        g[4 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 1], cos_v)
        g[5 + 4*kk] = dt_dense * np.dot(S_B_dense[:, 1], sin_v)
    return val, g


def speed_penalty_dense(m_fourier, t_param, K, omegas,
                         v_max_arg=None, zeta_speed_arg=None, n_dense=200):
    """Penalize sensor speed exceeding v_max."""
    if v_max_arg is None:
        v_max_arg = 0.5
    if zeta_speed_arg is None:
        zeta_speed_arg = 500.0
        
    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]
    
    vx, vy = fourier_velocity(m_fourier, t_dense, K, omegas)
    speed2 = vx**2 + vy**2
    v_max2 = v_max_arg**2
    
    excess = np.maximum(0.0, speed2 - v_max2)
    val = dt_dense * zeta_speed_arg * np.sum(excess**2)
    
    weight = 4.0 * zeta_speed_arg * excess
    
    g = np.zeros(4*K + 2)
    
    for kk in range(K):
        w = omegas[kk]
        sin_v = np.sin(w * t_dense)
        cos_v = np.cos(w * t_dense)
        
        dvx_dtheta = -w * sin_v
        dvx_dphi   =  w * cos_v
        dvy_dpsi   = -w * sin_v
        dvy_deta   =  w * cos_v
        
        g[2 + 4*kk] = dt_dense * np.dot(weight * vx, dvx_dtheta)
        g[3 + 4*kk] = dt_dense * np.dot(weight * vx, dvx_dphi)
        g[4 + 4*kk] = dt_dense * np.dot(weight * vy, dvy_dpsi)
        g[5 + 4*kk] = dt_dense * np.dot(weight * vy, dvy_deta)
    
    return val, g


def obstacle_penalty_dense(m_fourier, t_param, K, omegas,
                           obstacles, zeta_obs=2000.0, n_dense=200):
    """Penalize path entering obstacle + margin zone."""
    t_dense = np.linspace(t_param[0], t_param[-1], n_dense)
    dt_dense = t_dense[1] - t_dense[0]
    targets_dense = generate_targets(m_fourier, t_dense, K, omegas)
    
    val = 0.0
    S_obs = np.zeros((n_dense, 2))
    
    for obs in obstacles:
        margin = obs['margin']
        for j in range(n_dense):
            pt = targets_dense[j]
            dist, grad_dist = signed_distance_to_obstacle(pt, obs)
            
            penetration = margin - dist
            
            if penetration > 0:
                val += dt_dense * zeta_obs * penetration**2
                S_obs[j, 0] += -zeta_obs * 2.0 * penetration * grad_dist[0]
                S_obs[j, 1] += -zeta_obs * 2.0 * penetration * grad_dist[1]
    
    # Project onto Fourier
    g = np.zeros(4*K + 2)
    g[0] = dt_dense * np.sum(S_obs[:, 0])
    g[1] = dt_dense * np.sum(S_obs[:, 1])
    for kk in range(K):
        cos_v = np.cos(omegas[kk] * t_dense)
        sin_v = np.sin(omegas[kk] * t_dense)
        g[2 + 4*kk] = dt_dense * np.dot(S_obs[:, 0], cos_v)
        g[3 + 4*kk] = dt_dense * np.dot(S_obs[:, 0], sin_v)
        g[4 + 4*kk] = dt_dense * np.dot(S_obs[:, 1], cos_v)
        g[5 + 4*kk] = dt_dense * np.dot(S_obs[:, 1], sin_v)
    
    return val, g


def signed_distance_to_obstacle(xy, obs):
    """Signed distance from point xy to obstacle boundary."""
    x, y = xy[0], xy[1]
    
    if obs['type'] == 'circle':
        cx, cy = obs['center']
        r = obs['radius']
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist = d - r
        if d < 1e-12:
            grad_dist = np.array([1.0, 0.0])
        else:
            grad_dist = np.array([(x - cx) / d, (y - cy) / d])
        return dist, grad_dist
    
    elif obs['type'] == 'rectangle':
        xmin, ymin = obs['lower']
        xmax, ymax = obs['upper']
        
        dx = max(xmin - x, 0, x - xmax)
        dy = max(ymin - y, 0, y - ymax)
        
        if dx == 0 and dy == 0:
            # Inside rectangle
            dist = -min(x - xmin, xmax - x, y - ymin, ymax - y)
            dists_to_edges = [x - xmin, xmax - x, y - ymin, ymax - y]
            min_idx = np.argmin(dists_to_edges)
            grad_dist = [np.array([-1, 0]), np.array([1, 0]),
                         np.array([0, -1]), np.array([0, 1])][min_idx]
        else:
            dist = np.sqrt(dx**2 + dy**2)
            gx = 0.0; gy = 0.0
            if x < xmin:
                gx = (x - xmin)
            elif x > xmax:
                gx = (x - xmax)
            if y < ymin:
                gy = (y - ymin)
            elif y > ymax:
                gy = (y - ymax)
            norm = np.sqrt(gx**2 + gy**2)
            if norm < 1e-12:
                grad_dist = np.array([1.0, 0.0])
            else:
                grad_dist = np.array([gx / norm, gy / norm])
        
        return dist, grad_dist



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assemblePointwiseObservation(Vh, points):
    """Assemble pointwise observation matrix."""
    from hippylib import assemblePointwiseObservation as hippy_assemble
    return hippy_assemble(Vh, points)


def get_snapshot(traj, t, sim_times, Vh):
    """Get state snapshot at time t."""
    idx = int(np.argmin(np.abs(sim_times - t)))
    t_closest = float(sim_times[idx])
    assert abs(t_closest - t) < 1e-10, f"Time {t} not in simulation grid"
    snap = dl.Function(Vh).vector()
    traj.retrieve(snap, t_closest)
    return snap


def eval_fn_and_grad_P1(func, mesh, pt_xy):
    """Evaluate P1 function and its gradient at a point."""
    global _cached_bbt
    pt = dl.Point(float(pt_xy[0]), float(pt_xy[1]))
    val = func(pt)

    if _cached_bbt is None:
        _cached_bbt = mesh.bounding_box_tree()
    cid = _cached_bbt.compute_first_entity_collision(pt)
    if cid >= mesh.num_cells():
        return float(val), np.zeros(2)

    cell = dl.Cell(mesh, cid)

    try:
        coords = cell.get_vertex_coordinates().reshape(-1, 2)
    except AttributeError:
        verts = cell.entities(0)
        all_coords = mesh.coordinates()
        coords = all_coords[verts]

    dofs = func.function_space().dofmap().cell_dofs(cid)
    u_nod = np.array(func.vector()[dofs])

    A = np.column_stack([np.ones(3), coords])
    try:
        abc = np.linalg.solve(A, u_nod)
    except np.linalg.LinAlgError:
        return float(val), np.zeros(2)
    return float(val), abc[1:]


def reset_bbt_cache():
    """Reset the bounding box tree cache."""
    global _cached_bbt
    _cached_bbt = None

import sys
sys.path.append('../../../')
# Need to import these at the end to avoid circular imports
from hippylib import *# STATE, PARAMETER, ADJOINT, TimeDependentVector