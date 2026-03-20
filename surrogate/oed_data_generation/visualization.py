# """Visualization functions for OED training data generation."""

# import matplotlib.pyplot as plt
# import numpy as np
# import dolfin as dl
# import os

# from oed_core import generate_targets, fourier_velocity, get_snapshot
# from wind_sampler import sample_spectral_wind
# from hippylib import STATE


# def save_sample_visualization(sample_data, Vh, mesh, true_initial_condition,
#                              observation_times, simulation_times, K, omegas,
#                              obstacles, prior, eigsolver, r_modes,
#                              output_dir="sample_vis", dpi=150):
#     """
#     Save a comprehensive visualization of a single sample.
#     """
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Extract sample data
#     sample_idx = sample_data['sample_idx']
#     c_init = sample_data['c_init']
#     mean_vx = sample_data['mean_vx']
#     m_opt = sample_data['nn_output']
#     eig_opt = sample_data['eig_opt']
#     wind_coeffs = sample_data['wind_coeffs']
    
#     # Create initial path for comparison
#     m_init = create_initial_path(c_init, K)
    
#     # Build problems for EIG computation
#     prob_init, _, _ = build_problem_local(
#         m_init, Vh, prior, None,  # wind not needed for EIG
#         observation_times, simulation_times,
#         K, omegas, 1e-4
#     )
#     prob_opt, _, _ = build_problem_local(
#         m_opt, Vh, prior, None,
#         observation_times, simulation_times,
#         K, omegas, 1e-4
#     )
    
#     # Compute EIG values
#     _, _, eig_init = compute_eigendecomposition_local(
#         prob_init, prior, r_modes, eigsolver
#     )
    
#     # Create figure with subplots
#     fig = plt.figure(figsize=(20, 12))
    
#     # 1. Wind field and paths (top left)
#     ax1 = plt.subplot(2, 3, 1)
#     plot_wind_and_paths(ax1, mesh, wind_coeffs, m_init, m_opt, 
#                         observation_times, K, omegas, c_init)
#     ax1.set_title(f'Sample {sample_idx}: Wind Field & Sensor Paths\n'
#                   f'EIG: {eig_init:.2f} → {eig_opt:.2f}')
    
#     # 2. Concentration field (top middle)
#     ax2 = plt.subplot(2, 3, 2)
#     plot_concentration(ax2, Vh, true_initial_condition, m_opt, 
#                       observation_times, simulation_times, K, omegas)
#     ax2.set_title('Concentration Field at Final Time')
    
#     # 3. Obstacles and safety margins (top right)
#     ax3 = plt.subplot(2, 3, 3)
#     plot_obstacles(ax3, obstacles, m_opt, observation_times, K, omegas)
#     ax3.set_title('Obstacles & Safety Margins')
    
#     # 4. Speed profile (bottom left)
#     ax4 = plt.subplot(2, 3, 4)
#     plot_speed_profile(ax4, m_opt, observation_times, K, omegas, v_max=0.5)
#     ax4.set_title('Sensor Speed Profile')
    
#     # 5. Eigenvalue decay (bottom middle)
#     ax5 = plt.subplot(2, 3, 5)
#     plot_eigenvalues(ax5, prob_opt, prior, r_modes, eigsolver)
#     ax5.set_title('Hessian Eigenvalues')
    
#     # 6. Summary statistics (bottom right)
#     ax6 = plt.subplot(2, 3, 6)
#     plot_summary(ax6, sample_data, eig_init, eig_opt)
#     ax6.set_title('Sample Summary')
    
#     plt.tight_layout()
    
#     # Save figure
#     filename = os.path.join(output_dir, f'sample_{sample_idx:04d}_eig_{eig_opt:.2f}.png')
#     plt.savefig(filename, dpi=dpi, bbox_inches='tight')
#     plt.close(fig)
    
#     print(f"  Saved visualization to {filename}")


# def build_problem_local(m_fourier, Vh, prior, wind_velocity,
#                        observation_times, simulation_times,
#                        K, omegas, noise_variance):
#     """Local version of build_problem to avoid circular imports."""
#     from oed_core import generate_targets, MovingSensorMisfit
#     from model_ad_diff_bwd import TimeDependentAD
    
#     targets = generate_targets(m_fourier, observation_times, K, omegas)
#     misfit = MovingSensorMisfit(Vh, observation_times, targets)
    
#     problem = TimeDependentAD(
#         Vh.mesh(), [Vh, Vh, Vh], prior, misfit,
#         simulation_times, wind_velocity, True
#     )
#     return problem, misfit, targets


# def compute_eigendecomposition_local(prob, prior, r, eigsolver):
#     """Local version of compute_eigendecomposition."""
#     return eigsolver.solve(prob, prior, r)


# def plot_wind_and_paths(ax, mesh, wind_coeffs, m_init, m_opt, 
#                         observation_times, K, omegas, c_init):
#     """Plot wind field with initial and optimized paths."""
    
#     # Reconstruct wind field from coefficients
#     wind_field = reconstruct_wind_from_coeffs(mesh, wind_coeffs)
    
#     # Plot wind quiver on coarse grid
#     mc_q = dl.UnitSquareMesh(15, 15)
#     vc_q = dl.interpolate(wind_field, dl.VectorFunctionSpace(mc_q, "CG", 1))
#     coords_q = mc_q.coordinates()
#     vals_q = vc_q.compute_vertex_values(mc_q)
#     n_q = len(coords_q)
    
#     speed = np.sqrt(vals_q[:n_q]**2 + vals_q[n_q:]**2)
#     quiver = ax.quiver(coords_q[:, 0], coords_q[:, 1],
#                        vals_q[:n_q], vals_q[n_q:],
#                        speed, cmap='coolwarm', alpha=0.7, scale=20)
#     plt.colorbar(quiver, ax=ax, label='Wind speed')
    
#     # Generate paths
#     t_dense = np.linspace(observation_times[0], observation_times[-1], 200)
#     init_path = generate_targets(m_init, t_dense, K, omegas)
#     opt_path = generate_targets(m_opt, t_dense, K, omegas)
    
#     # Plot paths
#     ax.plot(init_path[:, 0], init_path[:, 1], 'r--', lw=2, alpha=0.7, 
#             label='Initial path')
#     ax.plot(opt_path[:, 0], opt_path[:, 1], 'b-', lw=3, alpha=0.9, 
#             label='Optimized path')
    
#     # Mark start and end
#     ax.scatter(opt_path[0, 0], opt_path[0, 1], c='lime', s=200, 
#                marker='*', edgecolors='black', zorder=5, label='Start')
#     ax.scatter(opt_path[-1, 0], opt_path[-1, 1], c='red', s=200, 
#                marker='s', edgecolors='black', zorder=5, label='End')
    
#     # Mark drone start position
#     ax.scatter(c_init[0], c_init[1], c='yellow', s=150, 
#                marker='D', edgecolors='black', zorder=5, label='Drone start')
    
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect('equal')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.legend(loc='upper right', fontsize=8)


# def plot_concentration(ax, Vh, true_ic, m_opt, observation_times, 
#                        simulation_times, K, omegas):
#     """Plot concentration field at final time."""
#     from model_ad_diff_bwd import TimeDependentAD
#     from oed_core import MovingSensorMisfit, generate_targets, get_snapshot
    
#     # Build problem
#     targets = generate_targets(m_opt, observation_times, K, omegas)
#     misfit = MovingSensorMisfit(Vh, observation_times, targets, 1e-4)
#     prob = TimeDependentAD(
#         Vh.mesh(), [Vh, Vh, Vh], None, misfit,
#         simulation_times, None, True
#     )
    
#     # Solve forward to get final state
#     u_final = prob.generate_vector(STATE)
#     x = [u_final, true_ic, None]
#     prob.solveFwd(u_final, x)
    
#     # Get final time snapshot
#     final_snap = get_snapshot(u_final, observation_times[-1], simulation_times, Vh)
#     final_func = dl.Function(Vh, final_snap)
    
#     # Plot
#     coords = Vh.tabulate_dof_coordinates()
#     values = final_func.vector().get_local()
#     contour = ax.tricontourf(coords[:, 0], coords[:, 1], values, 
#                              levels=20, cmap='viridis')
#     plt.colorbar(contour, ax=ax, label='Concentration')
    
#     # Overlay sensor positions
#     sensors = generate_targets(m_opt, observation_times, K, omegas)
#     ax.scatter(sensors[:, 0], sensors[:, 1], c='white', s=30, 
#                alpha=0.6, edgecolors='black', linewidths=0.5)
    
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect('equal')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')


# def plot_obstacles(ax, obstacles, m_opt, observation_times, K, omegas):
#     """Plot obstacles with safety margins and sensor path."""
    
#     # Plot obstacles
#     for obs in obstacles:
#         if obs['type'] == 'circle':
#             cx, cy = obs['center']
#             r = obs['radius']
#             m = obs['margin']
            
#             # Obstacle body
#             circle = plt.Circle((cx, cy), r, color='red', alpha=0.6, zorder=4)
#             ax.add_patch(circle)
            
#             # Safety margin
#             margin_circle = plt.Circle((cx, cy), r + m, color='red',
#                                         alpha=0.15, linestyle='--',
#                                         fill=True, zorder=3)
#             ax.add_patch(margin_circle)
            
#         elif obs['type'] == 'rectangle':
#             xmin, ymin = obs['lower']
#             xmax, ymax = obs['upper']
#             m = obs['margin']
#             w = xmax - xmin
#             h = ymax - ymin
            
#             # Obstacle body
#             rect = plt.Rectangle((xmin, ymin), w, h,
#                                   color='red', alpha=0.6, zorder=4)
#             ax.add_patch(rect)
            
#             # Safety margin
#             rect_m = plt.Rectangle((xmin - m, ymin - m), w + 2*m, h + 2*m,
#                                     color='red', alpha=0.15, linestyle='--',
#                                     fill=True, zorder=3)
#             ax.add_patch(rect_m)
    
#     # Plot sensor path
#     sensors = generate_targets(m_opt, observation_times, K, omegas)
#     ax.plot(sensors[:, 0], sensors[:, 1], 'b-', lw=2, alpha=0.8)
    
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_aspect('equal')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')


# def plot_speed_profile(ax, m_opt, observation_times, K, omegas, v_max=0.5):
#     """Plot sensor speed over time."""
    
#     t_dense = np.linspace(observation_times[0], observation_times[-1], 200)
#     vx, vy = fourier_velocity(m_opt, t_dense, K, omegas)
#     speed = np.sqrt(vx**2 + vy**2)
    
#     ax.plot(t_dense, speed, 'b-', lw=2, label='Speed')
#     ax.axhline(v_max, color='r', ls='--', alpha=0.7, label=f'v_max={v_max}')
#     ax.fill_between(t_dense, speed, v_max, where=(speed > v_max),
#                      color='red', alpha=0.3, label='Penalized')
    
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Speed |dc/dt|')
#     ax.legend()
#     ax.grid(True, alpha=0.3)


# def plot_eigenvalues(ax, prob, prior, r_modes, eigsolver):
#     """Plot Hessian eigenvalues."""
    
#     lmbda, _, _ = eigsolver.solve(prob, prior, r_modes)
    
#     ax.semilogy(range(1, len(lmbda)+1), lmbda, 'b*', markersize=8)
#     ax.axhline(1.0, color='r', ls='--', alpha=0.7, label='λ=1')
#     ax.set_xlabel('Mode index')
#     ax.set_ylabel('Eigenvalue λ')
#     ax.legend()
#     ax.grid(True, alpha=0.3)


# def plot_summary(ax, sample_data, eig_init, eig_opt):
#     """Plot summary statistics as text."""
    
#     ax.axis('off')
    
#     summary_text = f"""
#     Sample Information
#     ==================
#     Sample index: {sample_data['sample_idx']}
    
#     Drone start: ({sample_data['c_init'][0]:.3f}, {sample_data['c_init'][1]:.3f})
#     Mean wind vx: {sample_data['mean_vx']:.3f}
    
#     EIG initial: {eig_init:.2f}
#     EIG optimal: {eig_opt:.2f}
#     EIG gain: {eig_opt - eig_init:+.2f}
    
#     Optimization:
#     Converged: {sample_data['converged']}
#     Iterations: {sample_data['n_iter']}
#     Time: {sample_data['time']:.1f}s
    
#     Wind modes: {sample_data['wind_coeffs']['r_wind']}x{sample_data['wind_coeffs']['r_wind']}
#     Wind σ: {sample_data['wind_coeffs']['sigma']}
#     Wind α: {sample_data['wind_coeffs']['alpha']}
#     """
    
#     ax.text(0.1, 0.9, summary_text, fontsize=10, va='top',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# def reconstruct_wind_from_coeffs(mesh, coeffs):
#     """Reconstruct wind field from spectral coefficients."""
#     wind_field, _ = sample_spectral_wind(
#         mesh, r_wind=coeffs['r_wind'], sigma=coeffs['sigma'],
#         alpha=coeffs['alpha'], mean_vx=coeffs['mean_vx'],
#         mean_vy=coeffs['mean_vy'], seed=None
#     )
#     return wind_field


# def create_initial_path(c_init, K):
#     """Create initial Fourier path centered at drone position."""
#     m0 = np.zeros(4*K + 2)
#     m0[0] = c_init[0]
#     m0[1] = c_init[1]
#     m0[2] = 0.05
#     m0[5] = 0.05
#     return m0
"""Visualization functions for OED training data generation."""

import matplotlib.pyplot as plt
import numpy as np
import dolfin as dl
import os

from oed_core import generate_targets, fourier_velocity, get_snapshot
from wind_sampler import sample_spectral_wind
from hippylib import STATE


def save_sample_visualization(sample_data, Vh, mesh, true_initial_condition,
                             observation_times, simulation_times, K, omegas,
                             obstacles, prior, eigsolver, r_modes, wind_velocity,
                             output_dir="sample_vis", dpi=150):
    """
    Save a comprehensive visualization of a single sample.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract sample data
    sample_idx = sample_data['sample_idx']
    c_init = sample_data['c_init']
    mean_vx = sample_data['mean_vx']
    m_opt = sample_data['nn_output']
    eig_opt = sample_data['eig_opt']
    wind_coeffs = sample_data['wind_coeffs']
    
    # Create initial path for comparison
    m_init = create_initial_path(c_init, K)
    
    # Build problems for EIG computation - use the actual wind_velocity
    prob_init, _, _ = build_problem_local(
        m_init, Vh, prior, wind_velocity,
        observation_times, simulation_times,
        K, omegas, 1e-4
    )
    prob_opt, _, _ = build_problem_local(
        m_opt, Vh, prior, wind_velocity,
        observation_times, simulation_times,
        K, omegas, 1e-4
    )
    
    # Compute EIG values
    _, _, eig_init = compute_eigendecomposition_local(
        prob_init, prior, r_modes, eigsolver
    )
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Wind field and paths (top left)
    ax1 = plt.subplot(2, 3, 1)
    plot_wind_and_paths(ax1, mesh, wind_coeffs, m_init, m_opt, 
                        observation_times, K, omegas, c_init)
    ax1.set_title(f'Sample {sample_idx}: Wind Field & Sensor Paths\n'
                  f'EIG: {eig_init:.2f} → {eig_opt:.2f}')
    
    # 2. Concentration field (top middle)
    ax2 = plt.subplot(2, 3, 2)
    plot_concentration(ax2, Vh, true_initial_condition, m_opt, 
                      observation_times, simulation_times, K, omegas,
                      wind_velocity)  # Pass wind_velocity
    ax2.set_title('Concentration Field at Final Time')
    
    # 3. Obstacles and safety margins (top right)
    ax3 = plt.subplot(2, 3, 3)
    plot_obstacles(ax3, obstacles, m_opt, observation_times, K, omegas)
    ax3.set_title('Obstacles & Safety Margins')
    
    # 4. Speed profile (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    plot_speed_profile(ax4, m_opt, observation_times, K, omegas, v_max=0.5)
    ax4.set_title('Sensor Speed Profile')
    
    # 5. Eigenvalue decay (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    plot_eigenvalues(ax5, prob_opt, prior, r_modes, eigsolver)
    ax5.set_title('Hessian Eigenvalues')
    
    # 6. Summary statistics (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    plot_summary(ax6, sample_data, eig_init, eig_opt)
    ax6.set_title('Sample Summary')
    
    plt.tight_layout()
    
    # Save figure
    filename = os.path.join(output_dir, f'sample_{sample_idx:04d}_eig_{eig_opt:.2f}.png')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved visualization to {filename}")


def build_problem_local(m_fourier, Vh, prior, wind_velocity,
                       observation_times, simulation_times,
                       K, omegas, noise_variance):
    """Local version of build_problem to avoid circular imports."""
    from oed_core import generate_targets, MovingSensorMisfit
    from model_ad_diff_bwd import TimeDependentAD
    
    targets = generate_targets(m_fourier, observation_times, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets, noise_variance)
    
    problem = TimeDependentAD(
        Vh.mesh(), [Vh, Vh, Vh], prior, misfit,
        simulation_times, wind_velocity, True
    )
    return problem, misfit, targets


def compute_eigendecomposition_local(prob, prior, r, eigsolver):
    """Local version of compute_eigendecomposition."""
    return eigsolver.solve(prob, prior, r)


def plot_wind_and_paths(ax, mesh, wind_coeffs, m_init, m_opt, 
                        observation_times, K, omegas, c_init):
    """Plot wind field with initial and optimized paths."""
    
    # Reconstruct wind field from coefficients
    wind_field = reconstruct_wind_from_coeffs(mesh, wind_coeffs)
    
    # Plot wind quiver on coarse grid
    mc_q = dl.UnitSquareMesh(15, 15)
    vc_q = dl.interpolate(wind_field, dl.VectorFunctionSpace(mc_q, "CG", 1))
    coords_q = mc_q.coordinates()
    vals_q = vc_q.compute_vertex_values(mc_q)
    n_q = len(coords_q)
    
    speed = np.sqrt(vals_q[:n_q]**2 + vals_q[n_q:]**2)
    quiver = ax.quiver(coords_q[:, 0], coords_q[:, 1],
                       vals_q[:n_q], vals_q[n_q:],
                       speed, cmap='coolwarm', alpha=0.7, scale=20)
    plt.colorbar(quiver, ax=ax, label='Wind speed')
    
    # Generate paths
    t_dense = np.linspace(observation_times[0], observation_times[-1], 200)
    init_path = generate_targets(m_init, t_dense, K, omegas)
    opt_path = generate_targets(m_opt, t_dense, K, omegas)
    
    # Plot paths
    ax.plot(init_path[:, 0], init_path[:, 1], 'r--', lw=2, alpha=0.7, 
            label='Initial path')
    ax.plot(opt_path[:, 0], opt_path[:, 1], 'b-', lw=3, alpha=0.9, 
            label='Optimized path')
    
    # Mark start and end
    ax.scatter(opt_path[0, 0], opt_path[0, 1], c='lime', s=200, 
               marker='*', edgecolors='black', zorder=5, label='Start')
    ax.scatter(opt_path[-1, 0], opt_path[-1, 1], c='red', s=200, 
               marker='s', edgecolors='black', zorder=5, label='End')
    
    # Mark drone start position
    ax.scatter(c_init[0], c_init[1], c='yellow', s=150, 
               marker='D', edgecolors='black', zorder=5, label='Drone start')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)


def plot_concentration(ax, Vh, true_ic, m_opt, observation_times, 
                       simulation_times, K, omegas, wind_velocity):
    """Plot concentration field at final time."""
    from model_ad_diff_bwd import TimeDependentAD
    from oed_core import MovingSensorMisfit, generate_targets, get_snapshot
    
    # Build problem with actual wind_velocity
    targets = generate_targets(m_opt, observation_times, K, omegas)
    misfit = MovingSensorMisfit(Vh, observation_times, targets, 1e-4)
    prob = TimeDependentAD(
        Vh.mesh(), [Vh, Vh, Vh], None, misfit,
        simulation_times, wind_velocity, True  # Use actual wind_velocity
    )
    
    # Solve forward to get final state
    u_final = prob.generate_vector(STATE)
    x = [u_final, true_ic, None]
    prob.solveFwd(u_final, x)
    
    # Get final time snapshot
    final_snap = get_snapshot(u_final, observation_times[-1], simulation_times, Vh)
    final_func = dl.Function(Vh, final_snap)
    
    # Plot
    coords = Vh.tabulate_dof_coordinates()
    values = final_func.vector().get_local()
    contour = ax.tricontourf(coords[:, 0], coords[:, 1], values, 
                             levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Concentration')
    
    # Overlay sensor positions
    sensors = generate_targets(m_opt, observation_times, K, omegas)
    ax.scatter(sensors[:, 0], sensors[:, 1], c='white', s=30, 
               alpha=0.6, edgecolors='black', linewidths=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_obstacles(ax, obstacles, m_opt, observation_times, K, omegas):
    """Plot obstacles with safety margins and sensor path."""
    
    # Plot obstacles
    for obs in obstacles:
        if obs['type'] == 'circle':
            cx, cy = obs['center']
            r = obs['radius']
            m = obs['margin']
            
            # Obstacle body
            circle = plt.Circle((cx, cy), r, color='red', alpha=0.6, zorder=4)
            ax.add_patch(circle)
            
            # Safety margin
            margin_circle = plt.Circle((cx, cy), r + m, color='red',
                                        alpha=0.15, linestyle='--',
                                        fill=True, zorder=3)
            ax.add_patch(margin_circle)
            
        elif obs['type'] == 'rectangle':
            xmin, ymin = obs['lower']
            xmax, ymax = obs['upper']
            m = obs['margin']
            w = xmax - xmin
            h = ymax - ymin
            
            # Obstacle body
            rect = plt.Rectangle((xmin, ymin), w, h,
                                  color='red', alpha=0.6, zorder=4)
            ax.add_patch(rect)
            
            # Safety margin
            rect_m = plt.Rectangle((xmin - m, ymin - m), w + 2*m, h + 2*m,
                                    color='red', alpha=0.15, linestyle='--',
                                    fill=True, zorder=3)
            ax.add_patch(rect_m)
    
    # Plot sensor path
    sensors = generate_targets(m_opt, observation_times, K, omegas)
    ax.plot(sensors[:, 0], sensors[:, 1], 'b-', lw=2, alpha=0.8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_speed_profile(ax, m_opt, observation_times, K, omegas, v_max=0.5):
    """Plot sensor speed over time."""
    
    t_dense = np.linspace(observation_times[0], observation_times[-1], 200)
    vx, vy = fourier_velocity(m_opt, t_dense, K, omegas)
    speed = np.sqrt(vx**2 + vy**2)
    
    ax.plot(t_dense, speed, 'b-', lw=2, label='Speed')
    ax.axhline(v_max, color='r', ls='--', alpha=0.7, label=f'v_max={v_max}')
    ax.fill_between(t_dense, speed, v_max, where=(speed > v_max),
                     color='red', alpha=0.3, label='Penalized')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed |dc/dt|')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_eigenvalues(ax, prob, prior, r_modes, eigsolver):
    """Plot Hessian eigenvalues."""
    
    lmbda, _, _ = eigsolver.solve(prob, prior, r_modes)
    
    ax.semilogy(range(1, len(lmbda)+1), lmbda, 'b*', markersize=8)
    ax.axhline(1.0, color='r', ls='--', alpha=0.7, label='λ=1')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Eigenvalue λ')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_summary(ax, sample_data, eig_init, eig_opt):
    """Plot summary statistics as text."""
    
    ax.axis('off')
    
    summary_text = f"""
    Sample Information
    ==================
    Sample index: {sample_data['sample_idx']}
    
    Drone start: ({sample_data['c_init'][0]:.3f}, {sample_data['c_init'][1]:.3f})
    Mean wind vx: {sample_data['mean_vx']:.3f}
    
    EIG initial: {eig_init:.2f}
    EIG optimal: {eig_opt:.2f}
    EIG gain: {eig_opt - eig_init:+.2f}
    
    Optimization:
    Converged: {sample_data['converged']}
    Iterations: {sample_data['n_iter']}
    Time: {sample_data['time']:.1f}s
    
    Wind modes: {sample_data['wind_coeffs']['r_wind']}x{sample_data['wind_coeffs']['r_wind']}
    Wind σ: {sample_data['wind_coeffs']['sigma']}
    Wind α: {sample_data['wind_coeffs']['alpha']}
    """
    
    ax.text(0.1, 0.9, summary_text, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def reconstruct_wind_from_coeffs(mesh, coeffs):
    """Reconstruct wind field from spectral coefficients."""
    wind_field, _ = sample_spectral_wind(
        mesh, r_wind=coeffs['r_wind'], sigma=coeffs['sigma'],
        alpha=coeffs['alpha'], mean_vx=coeffs['mean_vx'],
        mean_vy=coeffs['mean_vy'], seed=None
    )
    return wind_field


def create_initial_path(c_init, K):
    """Create initial Fourier path centered at drone position."""
    m0 = np.zeros(4*K + 2)
    m0[0] = c_init[0]
    m0[1] = c_init[1]
    m0[2] = 0.05
    m0[5] = 0.05
    return m0