
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches
from utils import *
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
parser.add_argument('-rQ', '--rQ', type=int, default=202, help="rQ") #xvspectral input dimension, for now: 2 + 18 + 2
parser.add_argument('-dQ', '--dQ', type=int, default=14, help="dQ") # path coefficients
parser.add_argument('-rM', '--rM', type=int, default=100, help="rM")  
parser.add_argument('-dM', '--dM', type=int, default=1, help="dM") # eig
parser.add_argument('-data_type', '--data_type', type=str, default='eig', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=1600, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=100, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=7945, help="Max number of total data")
parser.add_argument('-plot_samples', '--plot_samples', type=int, default=4, help="Number of test samples to plot")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/', help="data directory")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/eig/', help="Directory to save models")
parser.add_argument('-epochs', '--epochs', type=int, default=200, help="epochs")
args = parser.parse_args()

batch_size = 32

assert args.n_train <= 8000 and args.n_train > 0

observation_times = np.linspace(T_1, T_FINAL, 50)  # Smooth path
omegas = fourier_frequencies(TY, K)
BUILDINGS = [
{'lower': (0.26, 0.16), 'upper': (0.49, 0.39), 'margin': 0.03},
{'lower': (0.61, 0.61), 'upper': (0.74, 0.84), 'margin': 0.03},
]
    
def draw_buildings(ax):
    """Draw building rectangles and margins on a plot axis."""
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h,
                                color='black', alpha=0.8, zorder=4)
        ax.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                    color='black', alpha=0.2, linestyle='--',
                                    fill=True, zorder=3)
        ax.add_patch(rect_m)



    # Draw buildings
    for b in BUILDINGS:
        xmin, ymin = b['lower']
        xmax, ymax = b['upper']
        w = xmax - xmin
        h = ymax - ymin
        m = b['margin']
        rect = mpatches.Rectangle((xmin, ymin), w, h,
                                color='black', alpha=0.8, zorder=4)
        ax1.add_patch(rect)
        rect_m = mpatches.Rectangle((xmin-m, ymin-m), w+2*m, h+2*m,
                                    color='black', alpha=0.2, linestyle='--',
                                    fill=True, zorder=3)
        ax1.add_patch(rect_m)
        

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
def combined_loss_func(model_output, target, input_q, lambda_consistency=.1):
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


mq_data_dict = np.load(args.data_dir+'mq_data_relabeled_pass2.npz')


m_data = mq_data_dict['m']
# in case mq_data has not been reduced, since this has been reduced via POD/active subspace, can just take the first rQ components
## I am NOT adding noise to the data here. Instead have a key with the noised data already, for efficiency and making sure all NNs are trained with the same type of noised data.
if args.data_type == 'xv':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v'] ), axis=1)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x'], mq_data_dict['v_mean'], mq_data_dict['v_coeff']), axis=1)
elif args.data_type == 'eig':
    q_data = mq_data_dict['m']
    m_data = mq_data_dict['eig_K3'] # take only the first dQ eig components as input

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

network, history = l2_training_standard(model,loss_func,train_invloader, validation_invloader,\
                     optimizer, lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose = True)

rel_error_test = evaluate_l2_error(model,test_invloader)

print('L2 relative error = ', rel_error_test)

model_save_name = f"eig_datatype_{args.data_type}_rQ{args.dQ}_rM{args.dM}_ntrain{args.n_train}.pth"
torch.save(model.state_dict(), os.path.join(args.data_dir, model_save_name))

plot_training_history('EIG', history, args.n_train, args.data_dir, args)

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

mse = mse_total / (n_batches*batch_size)
print(f'MSE loss = {mse:.6f}')

# Update the file writing to include MSE
with open(os.path.join(args.data_dir, "test_errors.txt"), "a") as f: 
    f.write(f"data_type={args.data_type}, n_train={args.n_train}, rel_error={rel_error_test}, mse={mse}\n")
