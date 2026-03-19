
import os, sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('../../')
sys.path.append('/workspace/arushi/hippylib')
sys.path.append('/workspace/arushi/hippyflow')

# https://github.com/dinoSciML/operator_learning_intro/tree/main/dinotorch_lite
from dinotorch_lite.src.dinotorch_lite import * 

from plotting.plot_trains import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-rQ', '--rQ', type=int, default=100, help="rQ")
parser.add_argument('-rM', '--rM', type=int, default=100, help="rM")
parser.add_argument('-data_type', '--data_type', type=int, default='xv', help="xv or xvspectral")
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")
parser.add_argument('-n_test', '--n_test', type=int, default=200, help="Number of test data")
parser.add_argument('-n_data', '--n_data', type=int, default=800, help="Max number of total data")
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/pointwise/', help="data directory")
parser.add_argument('-save_dir', '--save_dir', type=str, default='./models/rbno/', help="Directory to save models")
parser.add_argument('-epochs', '--epochs', type=int, default=1000, help="epochs")
args = parser.parse_args()

batch_size = 32

assert args.n_train <= 800 and args.n_train > 0



mq_data_dict = np.load(args.data_dir+'mq_data_reduced.npz')


m_data = mq_data_dict['m_data']
# in case mq_data has not been reduced, since this has been reduced via POD/active subspace, can just take the first rQ components
## I am NOT adding noise to the data here. Instead have a key with the noised data already, for efficiency and making sure all NNs are trained with the same type of noised data.
if args.data_type == 'xv':
    q_data = np.concatenate((mq_data_dict['x_data'], mq_data_dict['v_data']), axis=0)
elif args.data_type == 'xvspectral':
    q_data = np.concatenate((mq_data_dict['x_data'], mq_data_dict['v_coeff_data']), axis=0)


m_train = torch.Tensor(m_data[:args.n_train])
q_train = torch.Tensor(q_data[:args.n_train])

m_test = torch.Tensor(m_data[-args.n_test:])
q_test = torch.Tensor(q_data[-args.n_test:])

# Set up datasets and loaders
l2train = L2Dataset(m_train,q_train)
l2test = L2Dataset(m_test,q_test)
l2invtrain = L2Dataset(q_train,m_train)
l2invtest = L2Dataset(q_test,m_test)

train_loader = DataLoader(l2train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(l2test, batch_size=batch_size, shuffle=True)
train_invloader = DataLoader(l2invtrain, batch_size=batch_size, shuffle=True)
validation_invloader = DataLoader(l2invtest, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


################################################################################
# L2 training
model = GenericDense(input_dim = args.dM,hidden_layer_dim = 2*args.dM,output_dim = args.dQ).to(device)

n_epochs = 100
loss_func = normalized_f_mse
lr_scheduler = None

optimizer = torch.optim.Adam(model.parameters())

network, history = l2_training(model,loss_func,train_loader, validation_loader,\
                     optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose = True)

rel_error = evaluate_l2_error(model,validation_loader)

print('L2 relative error = ', rel_error)

model_save_name = f"rbno_datatype{args.data_type}_rQ{args.rQ}_rM{args.dM}_ntrain{args.n_train}.pth"
torch.save(model.state_dict(), os.path.join(args.data_dir, model_save_name))
plot_training_history('RBNO', history, args.n_train, args.data_dir, args)

