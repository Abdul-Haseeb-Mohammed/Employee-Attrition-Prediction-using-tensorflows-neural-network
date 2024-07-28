import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_lr_vs_loss(history, save_path=None):
    lrs = 1e-5 * (10 ** (np.arange(100) / 20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])  # x-axis in log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs. Loss")
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Learning rate vs. loss plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()

def plot_training_curves(history, model_name="Model", save_path=None):
    pd.DataFrame(history.history).plot()
    plt.title(f"{model_name} Training Curves")
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Training curves plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()
