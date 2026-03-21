import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
def plot_training_history(model_name, history, n_train, data_dir, args):
    """
    Plot training history with error handling.
    Handles different key naming conventions.
    """
    # Handle different possible key names
    train_loss_key = None
    val_loss_key = None
    epoch_key = None
    
    # Find train loss key
    for key in ['train_loss', 'train_loss_l2', 'train_loss_l1', 'training_loss']:
        if key in history:
            train_loss_key = key
            break
    
    # Find validation loss key
    for key in ['val_loss', 'validation_loss', 'validation_loss_l2', 'val_loss_l2']:
        if key in history:
            val_loss_key = key
            break
    
    # Find epoch key
    for key in ['epoch', 'epochs', 'iteration', 'iterations']:
        if key in history:
            epoch_key = key
            break
    
    # Extract data
    if train_loss_key:
        train_loss = history[train_loss_key]
        print(f"Found training loss key: {train_loss_key}, length: {len(train_loss)}")
    else:
        train_loss = []
        print("Warning: No training loss data found")
    
    if val_loss_key:
        val_loss = history[val_loss_key]
        print(f"Found validation loss key: {val_loss_key}, length: {len(val_loss)}")
    else:
        val_loss = []
        print("Warning: No validation loss data found")
    
    if epoch_key:
        epochs = history[epoch_key]
        print(f"Found epochs key: {epoch_key}, length: {len(epochs)}")
    else:
        # Create epochs based on loss length
        max_len = max(len(train_loss), len(val_loss))
        epochs = list(range(1, max_len + 1))
        print(f"Created epochs array of length {len(epochs)}")
    
    # Check if we have data to plot
    if len(train_loss) == 0 and len(val_loss) == 0:
        print("Warning: No training history to plot")
        return
    
    # Ensure epochs and losses have compatible lengths
    if len(train_loss) > 0 and len(epochs) != len(train_loss):
        print(f"Note: epochs length ({len(epochs)}) != train_loss length ({len(train_loss)})")
        # Use the shorter length
        min_len = min(len(epochs), len(train_loss))
        plot_epochs = epochs[:min_len]
        plot_train = train_loss[:min_len]
    else:
        plot_epochs = epochs[:len(train_loss)]
        plot_train = train_loss
    
    if len(val_loss) > 0 and len(epochs) != len(val_loss):
        print(f"Note: epochs length ({len(epochs)}) != val_loss length ({len(val_loss)})")
        min_len = min(len(epochs), len(val_loss))
        plot_val_epochs = epochs[:min_len]
        plot_val = val_loss[:min_len]
    else:
        plot_val_epochs = epochs[:len(val_loss)]
        plot_val = val_loss
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if len(plot_train) > 0:
        plt.plot(plot_epochs, plot_train, 'b-', label='Training Loss', linewidth=2, markersize=4)
        print(f"Plotted training loss: {len(plot_train)} points")
    
    # Plot validation loss
    if len(plot_val) > 0:
        plt.plot(plot_val_epochs, plot_val, 'r--', label='Validation Loss', linewidth=2, markersize=4)
        print(f"Plotted validation loss: {len(plot_val)} points")
    
    # Set labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} Training History (n_train={n_train})', fontsize=14)
    
    # Only add legend if we have labels
    if len(plot_train) > 0 or len(plot_val) > 0:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    # Use log scale if all losses are positive
    all_losses = []
    if len(plot_train) > 0:
        all_losses.extend(plot_train)
    if len(plot_val) > 0:
        all_losses.extend(plot_val)
    
    if len(all_losses) > 0 and np.all(np.array(all_losses) > 0):
        plt.yscale('log')
        print("Using log scale for y-axis")
    
    # Save plot
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f'{model_name}_ntrain_{n_train}_training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_training_history_multiple(histories, model_names, n_train, data_dir, args):
    """
    Plot multiple training histories on the same figure.
    
    Parameters
    ----------
    histories : list of dict
        List of history dictionaries
    model_names : list of str
        List of model names for legend
    n_train : int
        Number of training samples
    data_dir : str
        Directory to save the plot
    args : argparse.Namespace
        Command line arguments
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    for idx, (history, model_name) in enumerate(zip(histories, model_names)):
        # Find loss keys
        train_loss = None
        val_loss = None
        
        for key in ['train_loss', 'train_loss_l2']:
            if key in history:
                train_loss = history[key]
                break
        
        for key in ['val_loss', 'validation_loss', 'validation_loss_l2']:
            if key in history:
                val_loss = history[key]
                break
        
        if train_loss is not None:
            color = colors[idx % len(colors)]
            style = linestyles[idx % len(linestyles)]
            epochs = list(range(1, len(train_loss) + 1))
            plt.plot(epochs, train_loss, style, color=color, 
                    label=f'{model_name} (Train)', linewidth=2)
            
            if val_loss is not None:
                plt.plot(epochs[:len(val_loss)], val_loss, style, color=color, 
                        alpha=0.7, label=f'{model_name} (Val)', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training History Comparison (n_train={n_train})', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, f'training_history_comparison_ntrain_{n_train}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()


def plot_loss_curves(train_losses, val_losses, title='Training Curves', save_path=None):
    """
    Simple function to plot loss curves.
    
    Parameters
    ----------
    train_losses : list or array
        Training losses
    val_losses : list or array
        Validation losses
    title : str
        Plot title
    save_path : str or None
        Path to save the figure, if None, displays the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if np.all(np.array(train_losses) > 0):
        # plt.yscale('log')
        pass
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()