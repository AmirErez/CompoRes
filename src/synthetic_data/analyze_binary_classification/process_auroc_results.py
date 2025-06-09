import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_auroc_from_file(file_path: str):
    """
    Parses the AUROC value from a given results file.

    :param file_path: Path to the AUROC results file.
    :return: The AUROC value if found, otherwise None.
    """
    with open(file_path, 'r') as f:
        content = f.read()
        match = re.search(r"AUC:\s*([\d.]+)", content)
        if match:
            return float(match.group(1))
    return None


def collect_auroc_data(base_dir: str,
                       noise_levels_ocu_numbers: list[int],
                       sample_size: int,
                       noise_levels: list[float],
                       iterations: int,
                       coda_method: str,
                       response_based: str,
                       realizations: int,
                       exp_name: str) -> dict[tuple[int, int, float, int, str], list[float]]:
    """
    Collects AUROC data from results files and calculates the mean over realizations.

    :param base_dir: Directory containing results files.
    :param noise_levels_ocu_numbers: List of OCU numbers corresponding to the noise levels.
    :param sample_size: Number of samples in the dataset.
    :param noise_levels: List of noise levels.
    :param iterations: Number of iterations (number of responses generated of every type, correlated and uncorrelated).
    :param coda_method: Method for linear regression ('pairs' or 'CLR').
    :param response_based: Type of response ('balance' or 'taxon').
    :param realizations: Number of times the experiment was repeated.
    :param exp_name: Experiment name.
    :return: A dictionary with keys as tuples of (num_otus, sample_size, noise_level, iterations, coda_method) and
             values as the mean AUROC value for each combination of parameters.
    """
    noise_levels_ocu_numbers = [float(noise_ocu_number) for noise_ocu_number in noise_levels_ocu_numbers]
    auroc_values = {}
    for i, noise_level in enumerate(noise_levels):
        noise_level_ocu_number = int(noise_levels_ocu_numbers[i])
        key = (noise_level_ocu_number, sample_size, noise_level, iterations, coda_method)
        auroc_array = []
        for r in range(1, realizations + 1):
            dir_name = (
                f"{noise_level_ocu_number}otus_"
                f"{sample_size}samples_"
                f"{noise_level}noise_"
                f"{iterations}iterations_"
                f"{coda_method}_LR_method_"
                f"{response_based}_based_"
                f"realization{r}"
            )
            file_name = f"{exp_name}_auroc.txt"
            file_path = os.path.join(base_dir, dir_name, file_name)
            if os.path.exists(file_path):
                auroc = parse_auroc_from_file(file_path)
                if auroc is not None:
                    auroc_array.append(auroc)
            else:
               auroc_array.append(np.nan)
        if auroc_array:
            auroc_values[key] = auroc_array
    pd.DataFrame(auroc_values).to_csv(f"{base_dir}/auroc_data.csv")

    return auroc_values


def plot_auroc_vs_noise(auroc_values_dict: dict, realizations: int, response_based: str, coda_method: str,
                        number_of_responses: int, base_dir: str, plot_response_noise: bool,
                        exp_name: str, response_tag: str, response_rmse_values: list[float], sample_size: int):
    """Plots the mean AUROC vs. noise level.

    :param auroc_values_dict: AUROC values extracted using the `collect_auroc_data` method.
    :param realizations: Number of experiment repeats.
    :param response_based: The type of response ('balance' or 'taxon').
    :param coda_method: Method for linear regression ('pairs' or 'CLR').
    :param number_of_responses: Number of responses generated of every type, correlated and uncorrelated.
    :param base_dir: Directory to save the plot.
    :param plot_response_noise: Whether to plot responses noise levels range.
    :param exp_name: Microbiome case name.
    :param response_tag: Name of the response of interest.
    :param response_rmse_values: List of original response RMSE values.
    :param sample_size: Number of samples in the dataset.
    :return: None
    """

    mean_data = []
    std_data = []
    noise_levels = []
    noise_levels_ocu_numbers = []
    for key, auroc_array in auroc_values_dict.items():
        mean_data.append(float(np.nanmean(auroc_array)))
        std_data.append(float(np.nanstd(auroc_array)))
        noise_levels_ocu_numbers.append(int(key[0]))
        noise_levels.append(float(key[2]))

    # Create a figure with two vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex="all")

    # Plot AUROC vs. noise level
    ax1.plot(noise_levels, mean_data, color='#e69b00', label=f'Processed sample size {sample_size}')
    ax1.fill_between(noise_levels, np.array(mean_data) - np.array(std_data),
                         np.array(mean_data) + np.array(std_data), alpha=0.2, color='#e69b00', label=f'+- 1 SD')
    ax1.set_ylim(.5, 1)
    ax1.set_yticks(np.arange(.4, 1, 0.1))
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean AUROC')
    # calculate noise levels range
    noise_levels_range = np.max(noise_levels) - np.min(noise_levels)
    ax1.set_xlim(np.min(noise_levels) - noise_levels_range / 5, np.max(noise_levels) + noise_levels_range / 5)
    ax1.set_title(
        f'AUROC averaged across {realizations} realizations of {number_of_responses*2} responses vs. Noise level'
        f'\nMicrobiome: {exp_name}; Response: {response_tag}'
        f'\nLR transformation: OCU {coda_method} {response_based}'
    )
    ax1.legend()
    ax1.grid(True)

    if plot_response_noise:
        # Prepare data for the box plot
        values = [float(noise) for noise in response_rmse_values]

        # Create a twin y-axis to preserve the y-axis of the line plot
        ax2.boxplot(values, vert=False, boxprops=dict(color='blue'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    medianprops=dict(color='red'))
        ax2.set_yticks([])
        ax2.set_xlabel('RMSE Values')

        # Add scatter plot for the original response RMSE values
        ax2.scatter(values, np.ones(len(values)), color='blue', alpha=0.1, s=30)
        # Set the x-axis limits for both plots
        ax1.set_xlim(np.min(noise_levels) - noise_levels_range / 5, np.max(noise_levels) + noise_levels_range / 5)

    plt.tight_layout()

    if plot_response_noise:
        fig.savefig(f"{base_dir}/{exp_name}_mean_auroc_vs_noise_{coda_method}_{response_based}_with_noise_analysis.png")
        fig.savefig(f"{base_dir}/{exp_name}_mean_auroc_vs_noise_{coda_method}_{response_based}_with_noise_analysis.svg")
    else:
        fig.savefig(f"{base_dir}/{exp_name}_mean_auroc_vs_noise_{coda_method}_{response_based}.png")
        fig.savefig(f"{base_dir}/{exp_name}_mean_auroc_vs_noise_{coda_method}_{response_based}.svg")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--base_dir', type=str, default='../', help='Base directory')
    parser.add_argument('--noise_level_ocu_numbers', type=str, nargs='+', default=None, help='OCU numbers mapped to noise levels')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples in the dataset')
    parser.add_argument('--number_of_responses', type=int, default=50, help='Number of responses')
    parser.add_argument('--coda_method', type=str, default='CLR', help='')
    parser.add_argument('--realizations', type=int, default=10, help='Number of realizations')
    parser.add_argument('--exp_name', type=str, default='uncorrelated-IP', help='uncorrelated-{g2}')
    parser.add_argument('--response_tag', type=str, default='unknown', help='name of the response of interest')
    parser.add_argument('--response_based', type=str, default='balance', help='balance/taxon based')
    parser.add_argument('--plot_response_noise', type=bool, default=False, help='Plot responses noise levels range')
    parser.add_argument('--response_rmse_list', type=str, nargs='+', default=None, help='Deduplicated RMSE list')
    parser.add_argument('--response_initial_rmse_list', type=str, nargs='+', default=None, help='Initial RMSE list')

    args = parser.parse_args()
    print("Combining resulting mean ROC-AUC values vs. response noise levels visualization")
    data = collect_auroc_data(args.base_dir, args.noise_level_ocu_numbers, args.num_samples, args.response_rmse_list,
                              args.number_of_responses, args.coda_method, args.response_based, args.realizations,
                              args.exp_name)
    plot_auroc_vs_noise(data, args.realizations, args.response_based, args.coda_method, args.number_of_responses,
                        args.base_dir, args.plot_response_noise, args.exp_name, args.response_tag,
                        args.response_initial_rmse_list, args.num_samples)
