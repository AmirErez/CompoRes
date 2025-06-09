import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_p_value_frequencies(values_group1, name_g1, values_group2, name_g2, folder_to_save=""):
    # Convert lists to numpy arrays for processing and plotting
    values_group1 = np.array(values_group1, dtype=float)
    values_group2 = np.array(values_group2, dtype=float)

    # Calculate mean and standard deviation for Gaussian distribution
    mean_group1 = np.mean(values_group1)
    std_dev_group1 = np.std(values_group1)

    mean_group2 = np.mean(values_group2)
    std_dev_group2 = np.std(values_group2)

    # Plot the histograms
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(values_group1, bins=50, alpha=0.5, label=name_g1, density=True)
    ax.hist(values_group2, bins=50, alpha=0.5, label=name_g2, density=True)

    # Plot Gaussian distributions
    xmin, xmax = ax.get_xlim()
    x_group1 = np.linspace(xmin, xmax, 100)
    y_group1 = 1 / (std_dev_group1 * np.sqrt(2 * np.pi)) * np.exp(
        -(x_group1 - mean_group1) ** 2 / (2 * std_dev_group1 ** 2))
    ax.plot(x_group1, y_group1, label=f'{name_g1} Gaussian', linewidth=2)

    x_group2 = np.linspace(xmin, xmax, 100)  # Corrected line
    y_group2 = 1 / (std_dev_group2 * np.sqrt(2 * np.pi)) * np.exp(
        -(x_group2 - mean_group2) ** 2 / (2 * std_dev_group2 ** 2))
    ax.plot(x_group2, y_group2, label=f'{name_g2} Gaussian', linewidth=2)

    # Labeling the plot
    ax.set_xlabel('P-values')
    ax.set_ylabel('Density')
    ax.set_title('Frequency of p-values')
    ax.legend(loc='upper right')

    # Show and save the plot
    # fig.savefig(f"{folder_to_save}/p_value_frequencies.png")
    fig.savefig(f"{folder_to_save}/p_value_frequencies.svg")
    plt.close(fig)


def plot_p_value_boxplot(values_group1, name_g1, values_group2, name_g2, folder_to_save=""):
    # Convert lists to numpy arrays for processing and plotting
    values_group1 = np.array(values_group1, dtype=float)
    values_group2 = np.array(values_group2, dtype=float)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a DataFrame for Seaborn
    data = pd.DataFrame({
        'Group': [name_g1] * len(values_group1) + [name_g2] * len(values_group2),
        'Distance': np.concatenate([values_group1, values_group2])
    })

    # Create a box plot with individual data points
    sns.boxplot(x='Group', y='Distance', data=data, ax=ax, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "10"},
                boxprops=dict(facecolor="lightblue", edgecolor="blue"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                medianprops=dict(color="red"))

    sns.stripplot(x='Group', y='Distance', data=data, jitter=True, color='black', alpha=0.5, ax=ax)

    # Labeling the box plot
    ax.set_xlabel('Groups')
    ax.set_ylabel('P-values')
    # ax.set_yscale('log')
    ax.set_title('Box Plot of p-values with data points')

    # Show and save the plot
    # fig.savefig(f"{folder_to_save}/p_values_boxplot.png")
    fig.savefig(f"{folder_to_save}/p_values_boxplot.svg")
    plt.close(fig)


def read_p_values_from_dictionaries(correlated_p_value_dict_path, uncorrelated_p_value_dict_path, ocu_number):
    """Read p-values from dictionaries."""
    try:
        with open(correlated_p_value_dict_path, 'rb') as f:
            correlated_p_values = pickle.load(f)
    except FileNotFoundError:
        raise

    try:
        with open(uncorrelated_p_value_dict_path, 'rb') as f:
            uncorrelated_p_values = pickle.load(f)
    except FileNotFoundError:
        raise
    # correlated_distances, correlated_labels, uncorrelated_distances, uncorrelated_labels = [], [], [], []
    c_p_values_to_noise, correlated_labels, u_p_values_to_noise, uncorrelated_labels = [], [], [], []
    for data in [correlated_p_values, uncorrelated_p_values]:
        # Extracting the key and array
        for key in data:
            arr = data[key][ocu_number]
            # Checking if the key starts with "correlated" or "uncorrelated"
            if key.startswith("correlated"):
                correlated_labels = [1] * len(arr)
                # correlated_distances = [val[0] for val in arr.values()]
                c_p_values_to_noise = list(arr.values())
            elif key.startswith("uncorrelated"):
                uncorrelated_labels = [0] * len(arr)
                # uncorrelated_distances = [val[0] for val in arr.values()]
                u_p_values_to_noise = list(arr.values())
    # return correlated_distances, correlated_labels, uncorrelated_distances, uncorrelated_labels
    return c_p_values_to_noise, correlated_labels, u_p_values_to_noise, uncorrelated_labels


def classify_p_values(p_value, threshold):
    """Classify p-values as those of uncorrelated responses (0) if p_value >= threshold, else those correlated (1)."""
    classifications = [0 if d >= threshold else 1 for d in p_value]
    return classifications


def plot_roc_curve(correlated_p_values, correlated_labels, uncorrelated_p_values, uncorrelated_labels,
                   save_results_directory, exp_name):
    """Plot ROC curve for the given p-values, true labels, and thresholds."""

    auroc_path = os.path.join(save_results_directory, f"{exp_name}_auroc.txt")
    if os.path.exists(auroc_path):
        return
    p_values = np.concatenate((correlated_p_values, uncorrelated_p_values))
    true_labels = correlated_labels + uncorrelated_labels
    thresholds = np.linspace(np.min(p_values), np.max(p_values), 40)
    thresholds = np.concatenate(([-np.inf], thresholds, [np.inf]))
    plt.figure(figsize=(8, 6))
    all_fpr = []
    all_tpr = []

    for threshold in thresholds:
        classifications = classify_p_values(p_values, threshold)
        tn, fp, fn, tp = confusion_matrix(true_labels, classifications).ravel()
        # Calculate True Positive Rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Calculate False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    # Calculate AUROC
    auc = np.trapz(all_tpr, all_fpr)  # TODO: check when to move to trapezoid and whether to change the env

    with open(auroc_path, 'w') as f:
        f.write(f"AUC: {auc}")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    ax.plot(all_fpr, all_tpr, marker='o', linestyle='-', label='ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic Curve\n {exp_name}\nAUC: {auc:.2f}')
    ax.legend()
    ax.grid(True)
    fig.savefig(f"{save_results_directory}/roc_curve_{exp_name}.svg")
    plt.close(fig)
