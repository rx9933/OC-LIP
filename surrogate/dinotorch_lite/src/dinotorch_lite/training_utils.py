# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry

import torch, os
import numpy as np

def l2_training(model,loss_func,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False):
    device = next(model.parameters()).device

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss_l2'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u = batch
            m = m.to(device)
            u = u.to(device)
            u_pred = model(m)
            loss = loss_func(u_pred, u)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * m.shape[0]
            del u, m, u_pred
            torch.cuda.empty_cache()
        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            for batch in validation_loader:
                m, u = batch
                m = m.to(device)
                u = u.to(device)
                u_pred = model(m)
                # print(m.shape, u_pred.shape, u.shape)
                loss = loss_func(u_pred, u)
                validation_loss += loss.item() * m.shape[0]
                del u, m, u_pred
                torch.cuda.empty_cache()

        validation_loss /= len(validation_loader.dataset)
        train_history['validation_loss_l2'].append(validation_loss)
        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)
        if epoch %20 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")

    return model, train_history

def h1_training(model,loss_func_l2,loss_func_jac,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False,\
                     mode="forward", jac_weight = 1.0):
    device = next(model.parameters()).device

    def forward_pass(m):
        return model(torch.reshape(m, (-1, m.shape[-1])))

    if mode == "forward":
        jac_func = torch.func.vmap(torch.func.jacfwd(forward_pass))
    elif mode == "reverse":
        jac_func = torch.func.vmap(torch.func.jacrev(forward_pass))
    else:
        raise ValueError("Jacobian mode must be either 'forward' or 'reverse'")

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss'] = []
    train_history['validation_loss_l2'] = []
    train_history['validation_loss_jac'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u, J = batch
            m = m.to(device)
            u = u.to(device)
            J = J.to(device)
            u_pred = model(m)
            J_pred = jac_func(m)
            loss_l2 = loss_func_l2(u_pred, u)
            loss_jac = loss_func_jac(J_pred, J)
            loss = loss_l2 + jac_weight * loss_jac
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * m.shape[0]
            del u, m, J, u_pred
            torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            validation_loss_l2 = 0
            validation_loss_jac = 0
            for batch in validation_loader:
                m, u, J = batch
                m = m.to(device)
                u = u.to(device)
                J = J.to(device)
                u_pred = model(m)
                loss_l2 = loss_func_l2(u_pred, u)
                loss_jac = loss_func_jac(jac_func(m), J)
                loss = loss_l2 + jac_weight * loss_jac
                validation_loss += loss.item() * m.shape[0]
                validation_loss_l2 += loss_l2.item() * m.shape[0]
                validation_loss_jac += loss_jac.item() * m.shape[0]
                del u, m, J, u_pred
                torch.cuda.empty_cache()

        validation_loss /= len(validation_loader.dataset) 
        validation_loss_l2 /=len(validation_loader.dataset) 
        validation_loss_jac /= len(validation_loader.dataset) 

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)

        train_history['validation_loss'].append(validation_loss)
        train_history['validation_loss_l2'].append(validation_loss_l2)
        train_history['validation_loss_jac'].append(validation_loss_jac)

        # # Evaluation
        # with torch.no_grad():
        #     model.eval()
        #     validation_loss = 0
        #     for batch in validation_loader:
        #         m, u, J = batch
        #         m = m.to(device)
        #         u = u.to(device)
        #         u_pred = model(m)
        #         loss = loss_func(u_pred, u)
        #         validation_loss += loss.item() * m.shape[0]
        # validation_loss /= len(validation_loader.dataset)

        # # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)
        if epoch %10 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss L2: {validation_loss_l2:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss Jac: {validation_loss_jac:.6e}")

    return model, train_history

import torch
from torch.func import vmap, jacfwd, jacrev

# --- Helper function for L1 Regularization (LASSO) ---
def l1_regularization(model, l1_weight):
    """Calculates the L1 regularization term for a model's parameters."""
    l1_reg = torch.tensor(0., device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # Typically L1 is applied only to weights, not biases
        if 'weight' in name:
            l1_reg += torch.norm(param, p=1)
    return l1_weight * l1_reg

# ==============================================================================
# L2 Training with L1 Regularization (LASSO) and Early Stopping
# ==============================================================================
def l2_sparse_training(model, loss_func, train_loader, val_loader,
                       optimizer, lr_scheduler=None, n_epochs=100,
                       verbose=False, l1_weight=1e-5, patience=50):

    device = next(model.parameters()).device

    history = {
        "train_loss": [], 
        "validation_loss": [],
        "lr": [],
    }

    best_val = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):

        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        for m, u in train_loader:
            m, u = m.to(device), u.to(device)

            u_pred = model(m)
            data_loss = loss_func(u_pred, u)
            reg_loss = l1_regularization(model, l1_weight)

            loss = data_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * m.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for m, u in val_loader:
                m, u = m.to(device), u.to(device)
                u_pred = model(m)
                loss = loss_func(u_pred, u)
                val_loss += loss.item() * m.size(0)

        val_loss /= len(val_loader.dataset)

        # ===== EARLY STOPPING =====
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        # ===== LR SCHEDULER =====
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        # Save LR
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Log
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)

        if verbose and epoch % 20 == 0:
            print(f"[Epoch {epoch}] train={train_loss:.3e}, val={val_loss:.3e}, lr={history['lr'][-1]:.3e}")

    return model, history


# ==============================================================================
# H1 Training with L1 Regularization (LASSO) and Early Stopping
# ==============================================================================
def h1_sparse_training(model, loss_l2, loss_jac, train_loader, val_loader,
                       optimizer, lr_scheduler=None, n_epochs=100,
                       verbose=False, mode="reverse",
                       jac_weight=1.0, l1_weight=1e-5, patience=50):

    device = next(model.parameters()).device

    # Select Jacobian mode
    def fwd(x):
        return model(x.reshape(-1, x.shape[-1]))

    if mode == "forward":
        jac_func = vmap(jacfwd(fwd))
    else:
        jac_func = vmap(jacrev(fwd))

    history = {
        "train_loss": [],
        "train_loss_l2": [],
        "train_jac": [],
        "validation_loss": [],
        "validation_loss_l2": [],
        "validation_loss_jac": [],
        "lr": [],
    }

    best_val = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):

        # ===== TRAIN =====
        model.train()
        train_total = train_l2 = train_j = 0.0

        for m, u, J in train_loader:
            m, u, J = m.to(device), u.to(device), J.to(device)

            optimizer.zero_grad()

            u_pred = model(m)
            J_pred = jac_func(m)

            l2 = loss_l2(u_pred, u)
            jac = loss_jac(J_pred, J)
            reg = l1_regularization(model, l1_weight)

            loss = l2 + jac_weight * jac + reg

            loss.backward()
            optimizer.step()

            bs = m.size(0)
            train_total += loss.item() * bs
            train_l2 += l2.item() * bs
            train_j += jac.item() * bs

        n_train = len(train_loader.dataset)
        train_total /= n_train
        train_l2 /= n_train
        train_j /= n_train

        # ===== VALIDATION =====
        model.eval()
        val_total = val_l2 = val_j = 0.0

        with torch.no_grad():
            for m, u, J in val_loader:
                m, u, J = m.to(device), u.to(device), J.to(device)

                u_pred = model(m)
                J_pred = jac_func(m)

                l2 = loss_l2(u_pred, u)
                jac = loss_jac(J_pred, J)
                loss = l2 + jac_weight * jac

                bs = m.size(0)
                val_total += loss.item() * bs
                val_l2 += l2.item() * bs
                val_j += jac.item() * bs

        n_val = len(val_loader.dataset)
        val_total /= n_val
        val_l2 /= n_val
        val_j /= n_val

        # ===== EARLY STOPPING =====
        if val_total < best_val:
            best_val = val_total
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stop at epoch {epoch}")
                break

        # ===== LR UPDATE =====
        if lr_scheduler is not None:
            lr_scheduler.step(val_total)

        # Save LR
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Log
        history["train_loss"].append(train_total)
        history["train_loss_l2"].append(train_l2)
        history["train_jac"].append(train_j)
        history["validation_loss"].append(val_total)
        history["validation_loss_l2"].append(val_l2)
        history["validation_loss_jac"].append(val_j)

        if verbose and epoch % 10 == 0:
            print(f"[Epoch {epoch}] "
                  f"train={train_total:.3e} (l2={train_l2:.3e}, jac={train_j:.3e}) | "
                  f"val={val_total:.3e} (l2={val_l2:.3e}, jac={val_j:.3e}) | "
                  f"lr={history['lr'][-1]:.3e}")

    return model, history


def train_posa(model, train_dataloader, val_dataloader, 
               normalized_l2_loss, normalized_f_mse, 
               device, scale=1.0, lr=5e-4, n_epochs=100, 
               jac_mode="reverse", lr_scheduler=None, verbose=True):

    # Move model to device
    model.to(device)

    # Define forward pass for Jacobian computation
    def forward_pass(x):
        return model(torch.reshape(x, (-1, x.shape[-1])))

    # Choose Jacobian mode
    if jac_mode == "forward":
        jac_func = torch.func.vmap(torch.func.jacfwd(forward_pass))
    elif jac_mode == "reverse":
        jac_func = torch.func.vmap(torch.func.jacrev(forward_pass))
    else:
        raise ValueError("Jacobian mode must be either 'forward' or 'reverse'")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize loss history
    train_history = {
        'train_loss': [],
        'train_loss_l2': [],
        'train_loss_h1': [],
        'val_loss': [],
        'val_loss_l2': [],
        'val_loss_h1': []
    }

    for epoch in range(n_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        train_l2 = 0.0
        train_h1 = 0.0

        for batch in train_dataloader:
            inputs, targets, jacobians = batch
            inputs, targets, jacobians = inputs.to(device), targets.to(device), jacobians.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) / scale
            output_jacobians = jac_func(inputs) / scale

            l2_loss = normalized_l2_loss(outputs, targets)
            h1_loss = normalized_f_mse(output_jacobians, jacobians)
            loss = l2_loss + h1_loss

            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            train_loss += loss.item() * batch_size
            train_l2 += l2_loss.item() * batch_size
            train_h1 += h1_loss.item() * batch_size

        # Normalize training losses
        n_train = len(train_dataloader.dataset)
        train_loss /= n_train
        train_l2 /= n_train
        train_h1 /= n_train

        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        val_l2 = 0.0
        val_h1 = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                inputs, targets, jacobians = batch
                inputs, targets, jacobians = inputs.to(device), targets.to(device), jacobians.to(device)

                outputs = model(inputs) / scale
                output_jacobians = jac_func(inputs) / scale

                l2_loss = normalized_l2_loss(outputs, targets)
                h1_loss = normalized_f_mse(output_jacobians, jacobians)
                loss = l2_loss + h1_loss

                batch_size = inputs.shape[0]
                val_loss += loss.item() * batch_size
                val_l2 += l2_loss.item() * batch_size
                val_h1 += h1_loss.item() * batch_size

        # Normalize validation losses
        n_val = len(val_dataloader.dataset)
        val_loss /= n_val
        val_l2 /= n_val
        val_h1 /= n_val

        # Learning rate scheduling (if applicable)
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        # Store metrics
        train_history['train_loss'].append(train_loss)
        train_history['train_loss_l2'].append(train_l2)
        train_history['train_loss_h1'].append(train_h1)
        train_history['val_loss'].append(val_loss)
        train_history['val_loss_l2'].append(val_l2)
        train_history['val_loss_h1'].append(val_h1)

        # Print progress
        if verbose and epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train  | L2: {train_l2:.4e}, H1: {train_h1:.4e}, Total: {train_loss:.4e}")
            print(f"  Val    | L2: {val_l2:.4e}, H1: {val_h1:.4e}, Total: {val_loss:.4e}")

    return model, train_history




def load_all_data_with_jacobian(base_dir, n_data, low_rank):
    map0 = np.load(f"{base_dir}/data/map_param_0.npy")
    obs0 = np.load(f"{base_dir}/data/obs_0.npy")

    param_dim = map0.shape[0]
    obs_dim = obs0.shape[0]    

    map_param = np.zeros((n_data, param_dim))
    obs_data = np.zeros((n_data, obs_dim))
    posa_jac = np.zeros((n_data, param_dim, obs_dim))

    for i_data in range(n_data):
        map_param[i_data] = np.load(f"{base_dir}/data/map_param_{i_data}.npy")
        obs_data[i_data] = np.load(f"{base_dir}/data/obs_{i_data}.npy")

        if low_rank:
            posa_jac[i_data] = np.load(f"{base_dir}/data/low_rank_map_jacobian_{i_data}.npy")
        else:
            posa_jac[i_data] = np.load(f"{base_dir}/data/map_jacobian_{i_data}.npy")

    return map_param, obs_data, posa_jac 


def load_jobs_data(data_dir,  max_n=None, max_seed_limit=None, device='cpu'):
    """
    Load and directly reduce POSA data to Active Subspace dimensions on-the-fly.
    Returns reduced tensors: (map_param_reduced, obs_data_reduced, posa_jac_reduced)
    """
    print(f"\nLoading and reducing data from: {data_dir}")

    all_triplets = []  # temporarily hold all samples

    for seed_folder in sorted(os.listdir(data_dir)):
        if not seed_folder.startswith("seed"):
            continue

        try:
            seed = int(seed_folder.replace("seed", ""))
        except ValueError:
            continue

        if max_seed_limit is not None and seed > max_seed_limit:
            break

        folder_path = os.path.join(data_dir, seed_folder)
        print(f"Processing {seed_folder}...")

        file_groups = {}
        for fname in os.listdir(folder_path):
            if not fname.endswith(".npy"):
                continue
            try:
                index = int(fname.replace(".npy", "").split("_")[-1])
            except (ValueError, IndexError):
                index = 0

            if index not in file_groups:
                file_groups[index] = {}

            if "map_param" in fname:
                file_groups[index]["map_param"] = os.path.join(folder_path, fname)
            elif "obs" in fname:
                file_groups[index]["obs"] = os.path.join(folder_path, fname)
            elif "low_rank_map_jacobian_rank_400" in fname:
                file_groups[index]["posa_jac"] = os.path.join(folder_path, fname)

        for index in sorted(file_groups.keys()):
            fpaths = file_groups[index]
            if not all(k in fpaths for k in ["map_param", "obs", "posa_jac"]):
                print(f"  Missing files for index {index} in {seed_folder}, skipping.")
                continue

            try:
                # Load as float32 tensors
                m = torch.from_numpy(np.load(fpaths["map_param"])).float().to(device)
                d = torch.from_numpy(np.load(fpaths["obs"])).float().to(device)
                j = torch.from_numpy(np.load(fpaths["posa_jac"])).float().to(device)

                # --- Project to reduced space ---
                with torch.no_grad():
                    m_reduced = m.cpu().numpy()
                    j_reduced = j.reshape(j.shape[0], -1).T#,torch.einsum(
                        #"mr,rs->ms",
                        #j.reshape(j.shape[0], -1).T,
                        #AS_encoder_torch
                    #).T.cpu().numpy()
                    d_np = d.cpu().numpy()

                # store reduced triplet
                all_triplets.append((m_reduced, d_np, j_reduced))

                # free GPU memory early
                del m, d, j
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error loading/reducing index {index} in {seed_folder}: {e}")
                continue

            if max_n and len(all_triplets) >= max_n:
                break

    # Convert to arrays
    if all_triplets:
        map_params_r, obs_r, jac_r = zip(*all_triplets)
        map_params_r = np.array(map_params_r)
        obs_r = np.array(obs_r)
        jac_r = np.array(jac_r)
    else:
        map_params_r = np.empty((0,))
        obs_r = np.empty((0,))
        jac_r = np.empty((0,))

    print(f"\n✅ Reduced data shapes:")
    print(f"   map_param_reduced: {map_params_r.shape}")
    print(f"   obs_data_reduced:  {obs_r.shape}")
    print(f"   posa_jac_reduced:  {jac_r.shape}")

    return map_params_r, obs_r, jac_r
