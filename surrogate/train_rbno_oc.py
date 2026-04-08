
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'generate_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Setup finite element spaces for plotting
from generate_data.fe_setup import setup_fe_spaces
from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets
from generate_data.wind_utils import spectral_wind_to_field


# https://github.com/dinoSciML/operator_learning_intro/tree/main/dinotorch_lite
from dinotorch_lite.src.dinotorch_lite import * 
from plotting.plot_trains import *; from generate_data.config import * 
import dolfin as dl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-rQ', '--rQ', type=int, default=22, help="rQ") #xvspectral input dimension, for now: 2 + 18 + 2
parser.add_argument('-dQ', '--dQ', type=int, default=22, help="dQ")
parser.add_argument('-rM', '--rM', type=int, default=100, help="rM") 
parser.add_argument('-dM', '--dM', type=int, default=14, help="dM") # 14 path coefficients
parser.add_argument('-data_type', '--data_type', type=str, default='xvspectral', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=100, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=945, help="Max number of total data")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=4, help="Number of test samples to plot")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/rbno/', help="Directory to save models")
parser.add_argument('-epochs', '--epochs', type=int, default=1000, help="epochs")
args = parser.parse_args()

batch_size = 32

assert args.n_train <= 8000 and args.n_train > 0

observation_times = np.linspace(T_1, T_FINAL, 50)  # Smooth path
omegas = fourier_frequencies(TY, K)

def path_consistency_loss(model_output, input_q, K=3, omegas=omegas):
    """
    Compute consistency between input initial position and predicted path's initial position
    using the same generate_targets function as in plotting
    """
    # Extract initial position from input (first 2 dimensions)
    input_x0 = input_q[:, :2]  # [batch, 2]

    # Option 1: Using numpy-based generate_targets (exactly matches plotting)
    batch_size = model_output.shape[0]
    device = model_output.device
    
    predicted_x0_list = []
    
    for i in range(batch_size):
        coeffs_np = model_output[i].detach().cpu().numpy()
        # Use the exact same function from plotting
        path_np = generate_targets(coeffs_np, observation_times, K, omegas)
        
        # Extract initial position (first point)
        predicted_x0_np = path_np[0]  # [2]
        predicted_x0_list.append(predicted_x0_np)

    predicted_x0 = torch.from_numpy(np.array(predicted_x0_list)).float().to(device)
    consistency_loss = torch.mean((predicted_x0 - input_x0) ** 2)
    
    return consistency_loss

# path coeffs, target path coeffs, input data vector  
def combined_loss_func(model_output, target, input_q, lambda_consistency=10):
    """
    Returns: (total_loss, mse_loss, consistency_loss)
    """
    mse_loss = normalized_f_mse(model_output, target)
    consistency_loss = path_consistency_loss(model_output, input_q)
    
    # Optional: print every few batches for debugging
    # if np.random.random() < 0.01:
    #     print(f"  MSE: {mse_loss.item():.6f}, Consistency: {consistency_loss.item():.6f}")
    
    total_loss = mse_loss + lambda_consistency * consistency_loss
    
    return total_loss, mse_loss, lambda_consistency * consistency_loss


mq_data_dict = np.load(args.data_dir+'mq_data_reduced.npz')


m_data = mq_data_dict['m']
# in case mq_data has not been reduced, since this has been reduced via POD/active subspace, can just take the first rQ components
## I am NOT adding noise to the data here. Instead have a key with the noised data already, for efficiency and making sure all NNs are trained with the same type of noised data.
if args.data_type == 'xv':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v'] ), axis=1)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v_mean'], mq_data_dict['v_coeff']), axis=1)


m_train = torch.Tensor(m_data[:args.n_train])
q_train = torch.Tensor(q_data[:args.n_train])

m_val = torch.Tensor(m_data[args.n_train:-args.n_test])
q_val = torch.Tensor(q_data[args.n_train:-args.n_test])

m_test = torch.Tensor(m_data[-args.n_test:])
q_test = torch.Tensor(q_data[-args.n_test:])

# Set up datasets and loaders

l2invtrain = L2Dataset(q_train,m_train)
l2invval = L2Dataset(q_val,m_val)
l2invtest = L2Dataset(q_test,m_test)

train_invloader = DataLoader(l2invtrain, batch_size=batch_size, shuffle=True)
validation_invloader = DataLoader(l2invval, batch_size=batch_size, shuffle=True)
test_invloader = DataLoader(l2invtest, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

################################################################################
# L2 training
model = GenericDense(input_dim = args.dQ,hidden_layer_dim = 2*args.dQ,output_dim = args.dM).to(device)

n_epochs = args.epochs
loss_func = combined_loss_func
lr_scheduler = None

optimizer = torch.optim.Adam(model.parameters())

network, history = l2_training(model,loss_func,train_invloader, validation_invloader,\
                     optimizer, lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose = True, ic_penalty=True)

rel_error_test = evaluate_l2_error(model,test_invloader)

print('L2 relative error = ', rel_error_test)

model_save_name = f"rbno_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth"
torch.save(model.state_dict(), os.path.join(args.data_dir, model_save_name))

plot_training_history_with_consistency('RBNO', history, args.n_train, args.data_dir, args)

# Calculate MSE loss
model.eval()
mse_loss = torch.nn.MSELoss()
mse_total = 0.0
n_batches = 0

with torch.no_grad():
    for q_batch, m_batch in test_invloader:
        q_batch = q_batch.to(device)
        m_batch = m_batch.to(device)
        
        m_pred = model(q_batch)
        batch_mse = mse_loss(m_pred, m_batch)
        mse_total += batch_mse.item() * q_batch.size(0)
        n_batches += q_batch.size(0)

mse = mse_total / n_batches
print(f'MSE loss = {mse:.6f}')

# Update the file writing to include MSE
with open(os.path.join(args.data_dir, "test_errors.txt"), "a") as f: 
    f.write(f"data_type={args.data_type}, n_train={args.n_train}, rel_error={rel_error_test}, mse={mse}\n")
'''
# ============================================================================
# ADD THIS SECTION TO PLOT TEST EXAMPLES
# ============================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES")
print("="*60)


# Setup mesh and function spaces
# mesh, Vh_scalar, _ = setup_fe_spaces()
# V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
mesh = dl.UnitSquareMesh(NX, NY)
Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)  # For plotting
V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)  # For wind fields


# Get a few test samples
n_plot = min(args.plot_samples, len(m_test))
test_indices = np.arange(1, n_plot)#np.random.choice(len(m_test), n_plot, replace=False)

print(f"Plotting {n_plot} test examples...")

# Create a figure with multiple subplots for all test cases
fig = plt.figure(figsize=(20, 5*n_plot))
plot_idx = 1

for sample_idx in test_indices:
    # Get true data
    m_true = m_test[sample_idx].cpu().numpy()
    q_input = q_test[sample_idx].cpu().numpy()
    
    # Get corresponding clean data for wind field
    clean_idx = -args.n_test + sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % len(m_data)
    
    # Extract wind coefficients
    if args.data_type == 'xvspectral':
        x_init = q_input[:2]
        v_mean = q_input[2:4]
        v_coeff = q_input[4:]
        
        # Reconstruct wind coefficients dictionary
        r_wind = 3
        n_coeff_per_mode = r_wind * r_wind
        a_ij = v_coeff[:n_coeff_per_mode].reshape(r_wind, r_wind)
        b_ij = v_coeff[n_coeff_per_mode:].reshape(r_wind, r_wind)
        
        wind_coeffs = {
            'a_ij': a_ij,
            'b_ij': b_ij,
            'mean_vx': v_mean[0],
            'mean_vy': v_mean[1],
            'r_wind': r_wind,
            'sigma': 1.0,
            'alpha': 2.0
        }
        
        # Reconstruct wind field
        wind_field, _ = spectral_wind_to_field(mesh, wind_coeffs)
        
        # Get true EIG values
        eig_true = mq_data_dict['eig_opt'][clean_idx] # for spectral, was eig_opt
        eig_init = mq_data_dict['eig_init'][clean_idx] # for spectral, was eig_init 
        
    else:
        # For xv data type
        x_init = q_input[:2]
        v_coeff = q_input[2:]
        wind_coeffs = None
        wind_field = None
        eig_true = mq_data_dict['eig_K3'][clean_idx]
        eig_init = mq_data_dict['eig_K0'][clean_idx]
    
    # Predict with NN
    model.eval()
    with torch.no_grad():
        q_input_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
        m_pred = model(q_input_tensor).cpu().numpy().flatten()
    
    # Generate paths
    observation_times = np.linspace(T_1, T_FINAL, 50)  # Smooth path
    true_path = generate_targets(m_true, observation_times, K, omegas)
    pred_path = generate_targets(m_pred, observation_times, K, omegas)
    
    # Compute EIG for NN path (if you have this function)
    # For now, we'll use a placeholder or compute if available
    try:
        from oed_objective import compute_eig_for_path  # You need to implement this
        eig_pred = compute_eig_for_path(m_pred, wind_coeffs, mesh, Vh_scalar)
    except Exception as e:
        print(f"  WARNING: compute_eig_for_path failed: {e}")
        import traceback; traceback.print_exc()
        eig_pred = None
    if eig_pred is not None:
        from penalties import boundary_penalty_dense, speed_penalty_dense
        bdy_val, _ = boundary_penalty_dense(m_pred, observation_times, K, omegas)
        spd_val, _ = speed_penalty_dense(m_pred, observation_times, K, omegas)
        print(f"  NN penalties: boundary={bdy_val:.2f}, speed={spd_val:.2f}")
        print(f"  NN raw EIG={eig_pred:.4f}, NN penalized EIG={eig_pred - bdy_val - spd_val:.4f}")
        # Recompute true EIG with same fresh eigensolver for fair comparison
        eig_true_recomputed = compute_eig_for_path(m_true, wind_coeffs, mesh, Vh_scalar)
        print(f"  Stored PDE EIG={eig_true:.4f}, Recomputed PDE EIG={eig_true_recomputed:.4f}, NN EIG={eig_pred:.4f}")
    # Create subplot for this sample
    # Row 1: Wind field (if available) and paths
    ax1 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    
    if wind_field is not None:
        # Plot wind field with streamlines
        vx_func, vy_func = wind_field.split(deepcopy=True)
        coords = mesh.coordinates()
        vx_vals = vx_func.vector().get_local()
        vy_vals = vy_func.vector().get_local()
        
        # Create grid for plotting
        nx, ny = 40, 40
        xi = np.linspace(0, 1, nx)
        yi = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        Vx_grid = griddata((coords[:, 0], coords[:, 1]), vx_vals, (X, Y), method='linear')
        Vy_grid = griddata((coords[:, 0], coords[:, 1]), vy_vals, (X, Y), method='linear')
        
        speed = np.sqrt(Vx_grid**2 + Vy_grid**2)
        strm = ax1.streamplot(X, Y, Vx_grid, Vy_grid, color=speed, 
                             cmap='viridis', linewidth=1, density=1.2)
        plt.colorbar(strm.lines, ax=ax1, label='Speed (m/s)')
    else:
        ax1.text(0.5, 0.5, 'No wind field data', 
                ha='center', va='center', transform=ax1.transAxes)
    
    ax1.plot(x_init[0], x_init[1], 'go', markersize=8, label='Start')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.set_title(f'Sample {sample_idx}: Wind Field')
    ax1.grid(True, alpha=0.3)
    
    # Row 2: True vs Predicted Path
    ax2 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    ax2.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='True (PDE)')
    ax2.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='NN Prediction')
    ax2.plot(x_init[0], x_init[1], 'go', markersize=8, label='Start')
    ax2.plot(true_path[-1, 0], true_path[-1, 1], 'bs', markersize=8, label='True End')
    ax2.plot(pred_path[-1, 0], pred_path[-1, 1], 'rs', markersize=8, label='NN End')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.set_title(f'Sample {sample_idx}: Path Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Row 3: Path Error
    ax3 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    path_error = np.sqrt(np.sum((true_path - pred_path)**2, axis=1))
    times = observation_times - T_1
    ax3.plot(times, path_error, 'r-', linewidth=2)
    ax3.fill_between(times, 0, path_error, alpha=0.3, color='red')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error')
    ax3.set_title(f'Sample {sample_idx}: Path Error (RMSE: {np.mean(path_error):.4f})')
    ax3.grid(True, alpha=0.3)
    
    # Row 4: EIG Comparison
    ax4 = plt.subplot(n_plot, 4, plot_idx)
    plot_idx += 1
    
    labels = ['Initial', 'PDE Optimal']
    values = [eig_init, eig_true_recomputed]
    colors_bar = ['orange', 'blue']
    
    if eig_pred is not None:
        labels.append('NN Prediction')
        values.append(eig_pred)
        colors_bar.append('red')
    
    bars = ax4.bar(labels, values, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('EIG Value')
    ax4.set_title(f'Sample {sample_idx}: EIG Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    print(f"  Sample {sample_idx}: True EIG={eig_true:.4f}, Initial EIG={eig_init:.4f}, Path Error={np.mean(path_error):.4f}")

plt.tight_layout()

# Save the figure
plot_save_name = f"test_samples_{args.data_type}_ntrain{args.n_train}.png"
plot_save_path = os.path.join(args.data_dir, plot_save_name)
plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
print(f"\nTest samples plot saved to {plot_save_path}")
plt.close()

print("\n" + "="*60)
print("PLOTTING COMPLETE")
print("="*60)
'''

# ============================================================================
# MODIFIED PLOTTING SECTION - Similar to inspection script style
# ============================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES (Enhanced Visualization)")
print("="*60)

# Setup mesh and function spaces (same as inspection script)
MESH_FILE = 'ad_20.xml'
try:
    mesh = dl.refine(dl.Mesh(MESH_FILE))
    print(f"Loaded mesh from {MESH_FILE}")
except:
    print(f"Mesh file {MESH_FILE} not found, creating unit square mesh")
    mesh = dl.UnitSquareMesh(NX, NY)

Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)
V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)  # Use P2 for wind as in inspection script

# Create initial concentration field for background (same as inspection script)
def setup_initial_concentration(mesh):
    """Create initial concentration field (Gaussian blob)"""
    ic_expr = dl.Expression(
        'std::min(0.5, std::exp(-100*(std::pow(x[0]-0.35,2) + std::pow(x[1]-0.7,2))))',
        element=Vh_scalar.ufl_element()
    )
    true_ic = dl.interpolate(ic_expr, Vh_scalar)
    return true_ic

true_ic = setup_initial_concentration(mesh)

# Create regular grid for concentration visualization
x_vals = np.linspace(0, 1, 200)
y_vals = np.linspace(0, 1, 200)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# Evaluate concentration on grid
concentration_grid = np.zeros_like(X_grid)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        point = np.array([X_grid[i,j], Y_grid[i,j]])
        try:
            concentration_grid[i,j] = true_ic(point)
        except:
            concentration_grid[i,j] = 0.0

# Compute frequencies for path generation
omegas = fourier_frequencies(TY, K)

# Get a few test samples
n_plot = min(args.plot_samples, len(m_test))
# Use random samples instead of sequential for better variety
test_indices = np.random.choice(len(m_test), n_plot, replace=False)
print(f"Plotting {n_plot} test examples...")

# Create a figure with subplots for each test sample
# Each sample gets 2 rows x 2 cols = 4 subplots
fig = plt.figure(figsize=(18, 5*n_plot))

for plot_idx, sample_idx in enumerate(test_indices):
    print(f"\nProcessing sample {sample_idx}...")
    
    # Get data
    m_true = m_test[sample_idx].cpu().numpy()
    q_input = q_test[sample_idx].cpu().numpy()
    
    # Get corresponding clean data for wind field
    clean_idx = -args.n_test + sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % len(m_data)
    
    # Extract wind coefficients and initial position
    if args.data_type == 'xvspectral':
        x_init = q_input[:2]
        v_mean = q_input[2:4]
        v_coeff = q_input[4:]
        
        # Reconstruct wind coefficients dictionary
        r_wind = 3
        n_coeff_per_mode = r_wind * r_wind
        a_ij = v_coeff[:n_coeff_per_mode].reshape(r_wind, r_wind)
        b_ij = v_coeff[n_coeff_per_mode:].reshape(r_wind, r_wind)
        
        wind_coeffs = {
            'a_ij': a_ij,
            'b_ij': b_ij,
            'mean_vx': v_mean[0],
            'mean_vy': v_mean[1],
            'r_wind': r_wind,
            'sigma': 1.0,
            'alpha': 2.0
        }
        
        # Reconstruct wind field
        wind_field, _ = spectral_wind_to_field(mesh, wind_coeffs)
        
        # Get EIG values
        eig_true = mq_data_dict['eig_opt'][clean_idx]
        eig_init = mq_data_dict['eig_init'][clean_idx]
        
    else:
        x_init = q_input[:2]
        v_coeff = q_input[2:]
        wind_coeffs = None
        wind_field = None
        eig_true = mq_data_dict['eig_K3'][clean_idx]
        eig_init = mq_data_dict['eig_K0'][clean_idx]
    
    # Predict with NN
    model.eval()
    with torch.no_grad():
        q_input_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
        m_pred = model(q_input_tensor).cpu().numpy().flatten()
    
    # Generate dense paths for smooth visualization
    t_dense = np.linspace(T_1, T_FINAL, 200)
    true_path_dense = generate_targets(m_true, t_dense, K, omegas)
    pred_path_dense = generate_targets(m_pred, t_dense, K, omegas)
    
    # Generate sensor positions at observation times
    true_sensors = generate_targets(m_true, OBSERVATION_TIMES, K, omegas)
    pred_sensors = generate_targets(m_pred, OBSERVATION_TIMES, K, omegas)
    
    # Compute path error
    path_error = np.sqrt(np.sum((true_path_dense - pred_path_dense)**2, axis=1))
    
    # Create subplots for this sample (2 rows x 2 cols)
    # Row 1: Concentration background with paths (left) and Wind field with paths (right)
    ax1 = plt.subplot(n_plot, 2, 2*plot_idx + 1)
    ax2 = plt.subplot(n_plot, 2, 2*plot_idx + 2)
    
    # ===== LEFT PLOT: Concentration background with paths =====
    # Plot concentration field
    contour = ax1.contourf(X_grid, Y_grid, concentration_grid, 
                           levels=20, cmap='viridis', alpha=0.8)
    
    # Plot true and predicted paths
    ax1.plot(true_path_dense[:, 0], true_path_dense[:, 1], 'b-', lw=2.5, alpha=0.9,
             label=f'True (PDE) - EIG={eig_true:.1f}')
    ax1.plot(pred_path_dense[:, 0], pred_path_dense[:, 1], 'r--', lw=2, alpha=0.7,
             label=f'NN Prediction')
    
    # Plot sensor positions with time color map
    n_sensors = len(pred_sensors)
    sc1 = ax1.scatter(pred_sensors[:, 0], pred_sensors[:, 1], 
                     c=range(n_sensors), cmap='coolwarm',
                     s=40, alpha=0.8, edgecolors='black', linewidths=0.8, zorder=5,
                     label='Sensor positions')
    
    # Mark start and end positions
    ax1.scatter(x_init[0], x_init[1], 
               c='lime', s=120, marker='*', edgecolors='black', linewidths=1, zorder=6,
               label='Start position')
    ax1.scatter(pred_sensors[-1, 0], pred_sensors[-1, 1], 
               c='red', s=100, marker='s', edgecolors='black', linewidths=1, zorder=6,
               label='End position')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Sample {sample_idx}: Concentration Field & Paths')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for concentration
    plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04, label='Concentration')
    
    # ===== RIGHT PLOT: Wind field with paths =====
    if wind_field is not None:
        # Plot wind field using streamlines or quiver
        try:
            # Try streamplot first (looks nicer)
            vx_func, vy_func = wind_field.split(deepcopy=True)
            coords = mesh.coordinates()
            vx_vals = vx_func.vector().get_local()
            vy_vals = vy_func.vector().get_local()
            
            # Create grid for plotting
            nx, ny = 30, 30
            xi = np.linspace(0, 1, nx)
            yi = np.linspace(0, 1, ny)
            X_wind, Y_wind = np.meshgrid(xi, yi)
            
            from scipy.interpolate import griddata
            Vx_grid = griddata((coords[:, 0], coords[:, 1]), vx_vals, (X_wind, Y_wind), method='linear')
            Vy_grid = griddata((coords[:, 0], coords[:, 1]), vy_vals, (X_wind, Y_wind), method='linear')
            
            speed = np.sqrt(Vx_grid**2 + Vy_grid**2)
            strm = ax2.streamplot(X_wind, Y_wind, Vx_grid, Vy_grid, color=speed, 
                                 cmap='coolwarm', linewidth=1, density=1.2)
            plt.colorbar(strm.lines, ax=ax2, fraction=0.046, pad=0.04, label='Wind speed')
        except:
            # Fallback to quiver
            nx, ny = 20, 20
            xi = np.linspace(0, 1, nx)
            yi = np.linspace(0, 1, ny)
            X_wind, Y_wind = np.meshgrid(xi, yi)
            Vx_grid = griddata((coords[:, 0], coords[:, 1]), vx_vals, (X_wind, Y_wind), method='linear')
            Vy_grid = griddata((coords[:, 0], coords[:, 1]), vy_vals, (X_wind, Y_wind), method='linear')
            speed = np.sqrt(Vx_grid**2 + Vy_grid**2)
            quiver = ax2.quiver(X_wind, Y_wind, Vx_grid, Vy_grid, speed, 
                               cmap='coolwarm', alpha=0.7, scale=15, width=0.003)
            plt.colorbar(quiver, ax=ax2, fraction=0.046, pad=0.04, label='Wind speed')
    else:
        ax2.text(0.5, 0.5, 'No wind field data', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Plot paths on wind field
    ax2.plot(true_path_dense[:, 0], true_path_dense[:, 1], 'b-', lw=2.5, alpha=0.9, label='True path')
    ax2.plot(pred_path_dense[:, 0], pred_path_dense[:, 1], 'r--', lw=2, alpha=0.7, label='NN path')
    
    # Plot sensor positions
    sc2 = ax2.scatter(pred_sensors[:, 0], pred_sensors[:, 1], 
                     c=range(n_sensors), cmap='coolwarm',
                     s=40, alpha=0.8, edgecolors='black', linewidths=0.8, zorder=5)
    
    # Mark start position
    ax2.scatter(x_init[0], x_init[1], 
               c='lime', s=120, marker='*', edgecolors='black', linewidths=1, zorder=6)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Add wind statistics if available
    if args.data_type == 'xvspectral':
        ax2.set_title(f'Wind Field (vx_mean={v_mean[0]:.2f}, vy_mean={v_mean[1]:.2f})')
    else:
        ax2.set_title('Wind Field')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add text box with metrics
    rmse = np.mean(path_error)
    init_error = np.sqrt(np.sum((true_sensors[0] - pred_sensors[0])**2))
    end_error = np.sqrt(np.sum((true_sensors[-1] - pred_sensors[-1])**2))
    
    metrics_text = f'Path RMSE: {rmse:.4f}\nInitial error: {init_error:.4f}\nEnd error: {end_error:.4f}\nEIG gain: {eig_true - eig_init:.2f}'
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"  Sample {sample_idx}: Path RMSE={rmse:.4f}, Init Error={init_error:.4f}, End Error={end_error:.4f}")

plt.tight_layout()

# Save the figure
plot_save_name = f"test_samples_enhanced_{args.data_type}_ntrain{args.n_train}.png"
plot_save_path = os.path.join(args.data_dir, plot_save_name)
plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
print(f"\nEnhanced test samples plot saved to {plot_save_path}")

# Also create individual plots for each sample (like inspection script)
os.makedirs(os.path.join(args.data_dir, 'test_viz'), exist_ok=True)
for plot_idx, sample_idx in enumerate(test_indices):
    print(f"Saving individual plot for sample {sample_idx}...")
    
    # Re-extract data for this sample (reuse from above or recompute)
    m_true = m_test[sample_idx].cpu().numpy()
    q_input = q_test[sample_idx].cpu().numpy()
    
    clean_idx = -args.n_test + sample_idx
    if clean_idx < 0:
        clean_idx = clean_idx % len(m_data)
    
    if args.data_type == 'xvspectral':
        x_init = q_input[:2]
        v_mean = q_input[2:4]
        v_coeff = q_input[4:]
        
        r_wind = 3
        n_coeff_per_mode = r_wind * r_wind
        a_ij = v_coeff[:n_coeff_per_mode].reshape(r_wind, r_wind)
        b_ij = v_coeff[n_coeff_per_mode:].reshape(r_wind, r_wind)
        
        wind_coeffs = {
            'a_ij': a_ij,
            'b_ij': b_ij,
            'mean_vx': v_mean[0],
            'mean_vy': v_mean[1],
            'r_wind': r_wind,
            'sigma': 1.0,
            'alpha': 2.0
        }
        
        wind_field, _ = spectral_wind_to_field(mesh, wind_coeffs)
        eig_true = mq_data_dict['eig_opt'][clean_idx]
        eig_init = mq_data_dict['eig_init'][clean_idx]
    else:
        x_init = q_input[:2]
        wind_field = None
        eig_true = mq_data_dict['eig_K3'][clean_idx]
        eig_init = mq_data_dict['eig_K0'][clean_idx]
    
    with torch.no_grad():
        q_input_tensor = torch.FloatTensor(q_input).unsqueeze(0).to(device)
        m_pred = model(q_input_tensor).cpu().numpy().flatten()
    
    t_dense = np.linspace(T_1, T_FINAL, 200)
    true_path_dense = generate_targets(m_true, t_dense, K, omegas)
    pred_path_dense = generate_targets(m_pred, t_dense, K, omegas)
    
    # Create individual figure
    fig_ind, (ax1_ind, ax2_ind) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Concentration
    contour_ind = ax1_ind.contourf(X_grid, Y_grid, concentration_grid, levels=20, cmap='viridis', alpha=0.8)
    ax1_ind.plot(true_path_dense[:, 0], true_path_dense[:, 1], 'b-', lw=2.5, label='True path')
    ax1_ind.plot(pred_path_dense[:, 0], pred_path_dense[:, 1], 'r--', lw=2, label='NN prediction')
    ax1_ind.scatter(x_init[0], x_init[1], c='lime', s=120, marker='*', edgecolors='black', label='Start')
    ax1_ind.set_xlim(0, 1)
    ax1_ind.set_ylim(0, 1)
    ax1_ind.set_aspect('equal')
    ax1_ind.set_xlabel('x')
    ax1_ind.set_ylabel('y')
    ax1_ind.set_title(f'Sample {sample_idx}: Concentration Field')
    ax1_ind.legend()
    plt.colorbar(contour_ind, ax=ax1_ind, fraction=0.046, pad=0.04, label='Concentration')
    
    # Right: Wind field
    if wind_field is not None:
        vx_func, vy_func = wind_field.split(deepcopy=True)
        coords = mesh.coordinates()
        vx_vals = vx_func.vector().get_local()
        vy_vals = vy_func.vector().get_local()
        
        nx, ny = 30, 30
        xi = np.linspace(0, 1, nx)
        yi = np.linspace(0, 1, ny)
        X_wind, Y_wind = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        Vx_grid = griddata((coords[:, 0], coords[:, 1]), vx_vals, (X_wind, Y_wind), method='linear')
        Vy_grid = griddata((coords[:, 0], coords[:, 1]), vy_vals, (X_wind, Y_wind), method='linear')
        
        speed = np.sqrt(Vx_grid**2 + Vy_grid**2)
        strm_ind = ax2_ind.streamplot(X_wind, Y_wind, Vx_grid, Vy_grid, color=speed, 
                                     cmap='coolwarm', linewidth=1, density=1.2)
        plt.colorbar(strm_ind.lines, ax=ax2_ind, fraction=0.046, pad=0.04, label='Wind speed')
    else:
        ax2_ind.text(0.5, 0.5, 'No wind field', ha='center', va='center')
    
    ax2_ind.plot(true_path_dense[:, 0], true_path_dense[:, 1], 'b-', lw=2.5, label='True path')
    ax2_ind.plot(pred_path_dense[:, 0], pred_path_dense[:, 1], 'r--', lw=2, label='NN prediction')
    ax2_ind.scatter(x_init[0], x_init[1], c='lime', s=120, marker='*', edgecolors='black')
    ax2_ind.set_xlim(0, 1)
    ax2_ind.set_ylim(0, 1)
    ax2_ind.set_aspect('equal')
    ax2_ind.set_xlabel('x')
    ax2_ind.set_ylabel('y')
    ax2_ind.set_title(f'Sample {sample_idx}: Wind Field')
    ax2_ind.legend()
    
    # Add metrics text
    rmse_ind = np.mean(np.sqrt(np.sum((true_path_dense - pred_path_dense)**2, axis=1)))
    init_error_ind = np.sqrt(np.sum((true_path_dense[0] - pred_path_dense[0])**2))
    metrics_text_ind = f'Path RMSE: {rmse_ind:.4f}\nInit error: {init_error_ind:.4f}\nEIG: {eig_true:.2f} (gain: {eig_true-eig_init:.2f})'
    ax2_ind.text(0.02, 0.98, metrics_text_ind, transform=ax2_ind.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Test Sample {sample_idx}: True vs NN Predicted Path', fontsize=14)
    plt.tight_layout()
    
    # Save individual plot
    ind_save_path = os.path.join(args.data_dir, 'test_viz', f'test_sample_{sample_idx}.png')
    plt.savefig(ind_save_path, dpi=150, bbox_inches='tight')
    plt.close(fig_ind)

plt.close(fig)
print(f"\nIndividual test sample plots saved to {os.path.join(args.data_dir, 'test_viz')}/")
print("\n" + "="*60)
print("PLOTTING COMPLETE")
print("="*60)