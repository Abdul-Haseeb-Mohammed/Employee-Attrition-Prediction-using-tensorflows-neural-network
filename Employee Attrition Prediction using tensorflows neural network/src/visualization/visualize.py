import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


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

def metrics_score(actual, predicted, save_path=None):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Correlation matrix as a Heatmap saved to {os.path.abspath(save_path)}")
    else:
        plt.show()