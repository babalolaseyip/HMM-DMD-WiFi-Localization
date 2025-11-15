"""Visualization utilities for localization results."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_true[:, 0], y_true[:, 1], c='blue', alpha=0.5, label='True', s=50)
    axes[0].scatter(y_pred[:, 0], y_pred[:, 1], c='red', alpha=0.5, label='Predicted', s=50, marker='x')
    axes[0].set_xlabel('X coordinate (m)')
    axes[0].set_ylabel('Y coordinate (m)')
    axes[0].set_title('True vs Predicted Locations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Localization Error (m)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[2].plot(sorted_errors, cdf, linewidth=2)
    axes[2].set_xlabel('Localization Error (m)')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('Cumulative Distribution Function')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cdf_comparison(results_dict: dict, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    for method_name, (y_true, y_pred) in results_dict.items():
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cdf, linewidth=2, label=method_name)
    plt.xlabel('Localization Error (m)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('CDF Comparison of Localization Methods', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()
