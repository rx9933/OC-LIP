
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('../../')
sys.path.append('generate_data/')
sys.path.append('plotting/')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')

# https://github.com/dinoSciML/operator_learning_intro/tree/main/dinotorch_lite
from dinotorch_lite.src.dinotorch_lite import * 

from plotting.plot_trains import *
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

assert args.n_train <= 800 and args.n_train > 0



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
loss_func = normalized_f_mse
lr_scheduler = None

optimizer = torch.optim.Adam(model.parameters())

network, history = l2_training(model,loss_func,train_invloader, validation_invloader,\
                     optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose = True)

rel_error_test = evaluate_l2_error(model,test_invloader)

print('L2 relative error = ', rel_error_test)

model_save_name = f"rbno_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth"
torch.save(model.state_dict(), os.path.join(args.data_dir, model_save_name))

plot_training_history('RBNO', history, args.n_train, args.data_dir, args)

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

# ============================================================================
# ADD THIS SECTION TO PLOT TEST EXAMPLES
# ============================================================================
print("\n" + "="*60)
print("PLOTTING TEST EXAMPLES")
print("="*60)

# Setup finite element spaces for plotting
from generate_data.fe_setup import setup_fe_spaces
from generate_data.config import *
from fourier_utils import fourier_frequencies, generate_targets
from generate_data.wind_utils import spectral_wind_to_field

# Setup mesh and function spaces
# mesh, Vh_scalar, _ = setup_fe_spaces()
# V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
mesh = dl.UnitSquareMesh(NX, NY)
Vh_scalar = dl.FunctionSpace(mesh, 'Lagrange', 1)  # For plotting
V_vec = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)  # For wind fields


# Compute frequencies for path generation
omegas = fourier_frequencies(TY, K)

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
        eig_true = mq_data_dict['eig_opt'][clean_idx]
        eig_init = mq_data_dict['eig_init'][clean_idx]
        
    else:
        # For xv data type
        x_init = q_input[:2]
        v_coeff = q_input[2:]
        wind_coeffs = None
        wind_field = None
        eig_true = mq_data_dict['eig_opt'][clean_idx]
        eig_init = mq_data_dict['eig_init'][clean_idx]
    
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
    except:
        eig_pred = None
    
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
    values = [eig_init, eig_true]
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