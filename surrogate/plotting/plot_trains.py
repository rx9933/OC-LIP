import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
def plot_training_history(title, history, n_train, data_dir, args):
    """
    Plots the training history (loss vs. epoch) and saves the figure,
    handling history keys from both original and sparse training functions.
    """
    
    # 1. Setup the save directory
    # Assumes data_dir is something like 'data/pointwise/' and we want 'training_plots' 
    # at the same level as 'data' or inside the parent directory.
    save_dir = os.path.join(os.path.dirname(data_dir), 'training_plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. Determine plot data based on history keys (Robust Check)
    
    # Try getting the 'new' keys (from sparse training)
    train_loss = history.get('train_loss')
    val_loss = history.get('validation_loss')
    
    # If the new keys are None, fall back to the 'original' keys (from l2_training)
    if train_loss is None:
        train_loss = history.get('train_loss_l2')
    if val_loss is None:
        val_loss = history.get('validation_loss_l2')
    
    # Final check for data existence
    if train_loss is None or val_loss is None:
        print("Error: Required loss keys ('train_loss'/'train_loss_l2' or 'validation_loss'/'validation_loss_l2') not found in history.")
        return 
        
    # Now we are guaranteed to have valid lists for train_loss and val_loss
    epochs = range(1, len(train_loss) + 1)
    
    # Component losses for H1/DIED training (These use the newer keys)
    val_loss_l2_comp = history.get('validation_loss_l2')  # Use a different variable name to avoid confusion
    val_loss_jac = history.get('validation_loss_jac')

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the main training and validation losses
    # Note: If this came from l2_training, the label 'Total + L1' is misleading, 
    # but we stick to the provided labels for consistency with the new functions.
    plt.plot(epochs, train_loss, label='Train Loss (Total + L1)', color='blue', linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss (Total)', color='red', linestyle='--')

    # Plot H1 components if they exist (i.e., it was DIED/h1_sparse_training)
    # The presence of val_loss_jac indicates H1 training.
    if val_loss_jac is not None:
        # We plot val_loss_l2_comp because it represents the L2 component of the H1 validation loss.
        plt.plot(epochs, val_loss_l2_comp, label='Val Loss (L2 only)', color='orange', linestyle=':')
        plt.plot(epochs, val_loss_jac, label='Val Loss (Jacobian only)', color='green', linestyle=':')
        
        # Adjusting the title for DIED model
        # Assuming args.beta holds the jacobian weight (jac_weight)
        try:
            plot_title = f'{title} Training History (rQ={args.rQ}, rM={args.rM}, $\\beta$={args.beta:.0e})'
        except:
            plot_title = f'{title} Training History (dQ={args.dQ}, dM={args.dM}, $\\beta$={args.beta:.0e})'
    else:
        # Title for L2/ED model
        try:
            plot_title = f'{title} Training History (rQ={args.rQ}, rM={args.rM})'
        except:
            plot_title = f'{title} Training History (dQ={args.dQ}, dM={args.dM})'
    
    # 4. Set labels and title
    plt.title(plot_title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log') # Often helpful for training losses
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # 5. Save the figure
    # Using rQ and rM from args in the filename based on your script
    try:
        filename = f"{title.lower().replace(' ', '_')}_rQ{args.rQ}_rM{args.rM}_ntrain_{n_train}_beta{args.beta:.0e}.png"
    except:
        filename = f"{title.lower().replace(' ', '_')}_dQ{args.dQ}_dM{args.dM}_ntrain_{n_train}_beta{args.beta:.0e}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved successfully to: {save_path}")