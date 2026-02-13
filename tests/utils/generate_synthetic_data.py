import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import bisect

N_OTUS = 50
TARGET_N_EFF = 5
PERFECT_CORRELATION_PROBABILITY = 0.1
LOWER_CHOICE = -20
UPPER_CHOICE = 20
LOWER_CHOICE_PROBABILITY = 0.2
UPPER_CHOICE_PROBABILITY = 0.8
MEAN_ABUNDANCE = 0
MEAN_COVARIANCE = 1e-2
SAMPLE_SIZE = 100
RANDOM_STATE = 42
DEFAULT_PATH_TO_SIMULATED_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulated")
SIMULATED_CORRELATION_FILE_NAME = "synthetic_correlation_matrix"
SIMULATED_COVARIANCE_FILE_NAME = "synthetic_covariance_matrix"
SIMULATED_COMPOSITION_FILE_NAME = "synthetic_compositional_data"
SIMULATED_RESPONSE_FILE_NAME = "synthetic_response_data"
NUM_OF_RESPONSES_BY_TYPE = 5
RESPONSE_SLOPE_SCALER = 2
RESPONSE_INTERCEPT_SCALER = 5
BALANCE_OTU_LIST_LENGTH = 3
NOISE_LEVEL_LIST = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
RNG = np.random.default_rng(RANDOM_STATE)


def generate_random_sparse_correlation_matrices(
        n_otus: int = N_OTUS, prob: float = PERFECT_CORRELATION_PROBABILITY, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """
    Generate a symmetric sparse correlation matrices of size n_otus x n_otus.

    :param n_otus: Number of OTUs
    :param prob: Probability that a pair is perfectly correlated (+1 or -1)
    :param random_state: Seed
    :return: Positive-definite correlation matrix
    """
    rng = np.random.default_rng(random_state)
    mat = np.eye(n_otus)

    # Fill upper triangle (excluding diagonal)
    for i in range(n_otus):
        for j in range(i + 1, n_otus):
            if rng.uniform() > prob:
                value = 0.0
            else:
                value = rng.choice(
                    [LOWER_CHOICE, UPPER_CHOICE], p=[LOWER_CHOICE_PROBABILITY, UPPER_CHOICE_PROBABILITY]
                )  # asymmetric choice

            mat[i, j] = value
            mat[j, i] = value
    return mat

def nearest_positive_definite(input_mat: np.ndarray) -> np.ndarray:
    """Projects a given matrix to be the nearest positive-definite matrix.

    :param input_mat: Input matrix to adjust
    :return: Nearest positive-definite matrix
    """
    eigvals, eigvecs = np.linalg.eigh(input_mat)
    eigvals = np.clip(eigvals, 1e-6, None)
    mat_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(mat_pd))
    mat_pd = mat_pd / np.outer(d, d)
    return mat_pd

def compute_effective_number(proportions):
    proportions = proportions[proportions > 0]
    entropy = -np.sum(proportions * np.log(proportions))
    return np.exp(entropy)

def resolve_first_otu_mean_for_neff(
        n_otus: int = N_OTUS, target_neff: int = TARGET_N_EFF, common_mean: float = MEAN_ABUNDANCE) -> float:
    """
    Find mean value for the first OTU so that the effective number matches the target value of effective OUT number.

    :param n_otus: Total number of OTUs
    :param target_neff: Target effective number of OTUs
    :param common_mean: Mean log-abundance for all OTUs except the first
    :param common_var: Variance of log-abundance for all OTUs except the first
    """
    def neff_given_first_otu_mean(first_otu_mean):
        mu = np.full(n_otus, common_mean)
        mu[0] = first_otu_mean
        abund = np.exp(mu)
        props = abund / abund.sum()
        return compute_effective_number(props) - target_neff

    # Search in a reasonable range
    first_otu_mean = bisect(neff_given_first_otu_mean, 0, 80)
    return first_otu_mean

def cast_correlation_to_covariance(corr_matrix: np.ndarray, variance: float = MEAN_COVARIANCE) -> np.ndarray:
    """Converts a correlation matrix to a covariance matrix with a specified variance.

    :param corr_matrix: Correlation matrix to convert
    :param variance: Desired variance for the diagonal elements of the covariance matrix
    """
    std_vec = np.full(corr_matrix.shape[0], np.sqrt(variance))
    D = np.diag(std_vec)
    return D @ corr_matrix @ D

def set_up_compositional_data_paths(result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA):
    file_name_suffix = set_up_path_suffix()
    corr_file_name = f"{SIMULATED_CORRELATION_FILE_NAME}{file_name_suffix}.tsv"
    corr_file_name_path = os.path.join(result_dir, corr_file_name)
    cov_file_name = f"{SIMULATED_COVARIANCE_FILE_NAME}{file_name_suffix}.tsv"
    cov_file_name_path = os.path.join(result_dir, cov_file_name)
    compositional_data_file_name = f"{SIMULATED_COMPOSITION_FILE_NAME}{file_name_suffix}.tsv"
    compositional_data_file_name_path = os.path.join(result_dir, compositional_data_file_name)
    return corr_file_name_path, cov_file_name_path, compositional_data_file_name_path

def set_up_response_data_path(
        noise_value: float, balance_method: str, result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA
):
    file_name_suffix = set_up_path_suffix()
    response_file_name_suffix = f"_{str(noise_value)}_noise_{balance_method}"
    response_data_file_name = f"{SIMULATED_RESPONSE_FILE_NAME}{file_name_suffix}{response_file_name_suffix}.tsv"
    response_data_file_name_path = os.path.join(result_dir, response_data_file_name)
    return response_data_file_name_path


def set_up_path_suffix():
    sample_params = f"_{N_OTUS}_otus_{TARGET_N_EFF}_neff_{SAMPLE_SIZE}_samples_{RANDOM_STATE}_seed"
    prob = f"_{PERFECT_CORRELATION_PROBABILITY}_corr_prob"
    sparsity = f"_{LOWER_CHOICE}_{LOWER_CHOICE_PROBABILITY}_{UPPER_CHOICE}_{UPPER_CHOICE_PROBABILITY}_sparsity_params"
    mean_values = f"_{MEAN_ABUNDANCE}_mean_{MEAN_COVARIANCE}_cov"
    file_name_suffix = f"{sample_params}{prob}{sparsity}{mean_values}"
    return file_name_suffix


def simulate_microbiome_data(result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA) -> tuple:

    os.makedirs(result_dir, exist_ok=True)
    corr_file_name_path, cov_file_name_path, data_file_name_path = set_up_compositional_data_paths(result_dir)
    data_file_name, data_file_extension = os.path.splitext(data_file_name_path)
    abs_file_name_path = f"{data_file_name}_counts{data_file_extension}"

    # Check if files already exist, read them and skip generation if they do
    if os.path.exists(data_file_name_path):
        corr_matrix = pd.read_csv(corr_file_name_path, sep="\t", index_col=0)
        cov_matrix = pd.read_csv(cov_file_name_path, sep="\t", index_col=0)
        compositions = pd.read_csv(data_file_name_path, sep="\t", index_col=0)
        print(f"Using existing simulated microbiome data from {result_dir}")

    else:
        corr_matrix = generate_random_sparse_correlation_matrices()
        corr_matrix = nearest_positive_definite(corr_matrix)
        first_otu_mean_value = resolve_first_otu_mean_for_neff()
        cov_matrix = cast_correlation_to_covariance(corr_matrix)
        mu = np.full(N_OTUS, MEAN_ABUNDANCE)
        mu[0] = first_otu_mean_value
        log_abundances = np.random.multivariate_normal(mu, cov_matrix, SAMPLE_SIZE)
        abs_abundances = np.exp(log_abundances)
        compositions = abs_abundances / abs_abundances.sum(axis=1, keepdims=True)

        otu_labels = [f"OTU_{i + 1}" for i in range(N_OTUS)]
        sample_names = [f"Sample_{i + 1}" for i in range(SAMPLE_SIZE)]
        pd.DataFrame(corr_matrix, columns=otu_labels, index=otu_labels).to_csv(corr_file_name_path, sep="\t")
        pd.DataFrame(cov_matrix, columns=otu_labels, index=otu_labels).to_csv(cov_file_name_path, sep="\t")
        pd.DataFrame(compositions, columns=otu_labels, index=sample_names).to_csv(data_file_name_path, sep="\t")
        pd.DataFrame(abs_abundances, columns=otu_labels, index=sample_names).to_csv(abs_file_name_path, sep="\t")

    return corr_matrix, cov_matrix, compositions

def create_response_series(
        noise_value: float, response_name: str,
        result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA, correlation: str = "correlated", base: str = "pairs"
    ) -> tuple[pd.Series, dict[str, str | list[str]]]:
    """This function generates a synthetic response by adding Gaussian noise to computed values
    derived from microbiome data, depending on the specified correlation type and response basis.


    :param noise_value: The standard deviation of the Gaussian noise to be added.
    :param response_name: Name of the response variable.
    :param result_dir: Directory path where the results will be saved.
    :param correlation: Specifies if the response should be "correlated" or "uncorrelated".
    :param base: Determines if the response is based on "pairs" or "CLR".
    :return: A Pandas Series containing the generated response values.
    """
    response_update = {}
    corr_file_name_path, _, composition_data_file_name_path = set_up_compositional_data_paths(result_dir)

    # Keep a scaled noise for uncorrelated responses and for final additive noise
    noise_array_for_additive = noise_value * RNG.standard_normal(SAMPLE_SIZE)
    # Pass a unit (unscaled) noise vector to the correlation in the generate_linear_model
    unit_noise_for_corr = RNG.standard_normal(SAMPLE_SIZE)
    sample_names = [f"Sample_{i + 1}" for i in range(SAMPLE_SIZE)]
    response = np.zeros(SAMPLE_SIZE)

    response_update["Response_Name"] = response_name

    if correlation == "correlated":
        corr_matrix = pd.read_csv(corr_file_name_path, sep="\t", index_col=0)
        samples = pd.read_csv(composition_data_file_name_path, sep="\t", index_col=0)
        pos_otus, neg_otus = select_correlated_otus(
            corr_matrix, BALANCE_OTU_LIST_LENGTH
        )
        response_update["NUM"] = pos_otus
        if base == "pairs":
            response_update["DEN"] = neg_otus

        elif base == "CLR":
            # Update neg_otus to be all OTUs not in pos_otus to use the existing helper function
            neg_otus = [otu for otu in samples.columns if otu not in pos_otus]
            response_update["DEN"] = "CLR"

        # Compute balance
        balance_used = calculate_irl_balance_from_groups(samples, pos_otus, neg_otus)
        response_base, response_update = generate_linear_model(balance_used, unit_noise_for_corr, response_update)
        # add scaled noise_value to degrade the signal according to noise_value as intended
        response = response_base + noise_array_for_additive

    elif correlation == "uncorrelated":
        response = noise_array_for_additive

    response = pd.Series(response, name=response_name)
    response.index = sample_names

    return response, response_update


def calculate_irl_balance_from_groups(
        samples: pd.DataFrame, pos_cols: list, neg_cols: list, pseudocount: float = 1e-9
) -> np.ndarray:
    """ILR-style balance between two groups: coef * log(gmean(pos) / gmean(neg))
    samples: DataFrame (samples x taxa), may be proportions or counts
    """
    X = samples.loc[:, pos_cols + neg_cols].astype(float) + pseudocount
    # geometric means per sample
    gmean_pos = np.exp(np.log(X.loc[:, pos_cols]).mean(axis=1))
    gmean_neg = np.exp(np.log(X.loc[:, neg_cols]).mean(axis=1))
    k_pos = len(pos_cols)
    k_neg = len(neg_cols)
    coef = np.sqrt((k_pos * k_neg) / (k_pos + k_neg)) if (k_pos + k_neg) > 0 else 1.0
    balance = coef * np.log(gmean_pos / gmean_neg)
    return balance.to_numpy()


def select_correlated_otus(corr_matrix: pd.DataFrame, list_length: int) -> tuple[list, list]:
    """
    Deterministic selection:
    - pick OTUs appearing in the pairs with highest positive correlations (preserve order)
    - for negatives: for each pos otu, pick most negative correlated OTUs
    """
    mat = corr_matrix.values.copy()
    np.fill_diagonal(mat, 0.0)
    n = mat.shape[0]
    # get pair indices sorted by correlation descending
    flat_idx = np.argsort(mat.flatten())[::-1]
    pos_otus = []
    seen = set()
    for idx in flat_idx:
        i, j = divmod(int(idx), n)
        if i == j:
            continue
        for k in (i, j):
            label = str(corr_matrix.index[k])
            if label not in seen:
                pos_otus.append(label)
                seen.add(label)
                if len(pos_otus) >= list_length:
                    break
        if len(pos_otus) >= list_length:
            break

    # find most negative partners for pos_otus
    neg_otus = []
    seen_neg = set()
    for pos in pos_otus:
        row = corr_matrix.loc[pos, :]
        negs = row[row < 0].sort_values()  # ascending: most negative first
        for idx in negs.index:
            if idx not in seen_neg and idx not in pos_otus:
                neg_otus.append(str(idx))
                seen_neg.add(idx)
            if len(neg_otus) >= list_length:
                break
        if len(neg_otus) >= list_length:
            break

    # final trim
    pos_otus = pos_otus[:list_length]
    neg_otus = neg_otus[:list_length]
    return pos_otus, neg_otus


def generate_linear_model(balance_used: np.ndarray, noise_array: np.ndarray, response_update: dict):
    # Desired Pearson correlation between balance_used and response
    target_corr = np.random.uniform(0.85, 0.95)
    response_update["TARGET_CORR"] = target_corr

    b_mean = np.mean(balance_used)
    b_std = np.std(balance_used, ddof=0)
    if b_std == 0 or np.isnan(b_std):
        b_std = 1.0
    balance_std = (balance_used - b_mean) / b_std

    n_mean = np.mean(noise_array)
    n_std = np.std(noise_array, ddof=0)
    if n_std == 0 or np.isnan(n_std):
        n_std = 1.0
    noise_std = (noise_array - n_mean) / n_std

    r = float(np.clip(target_corr, -0.9999, 0.9999))
    sqrt_term = np.sqrt(max(0.0, 1.0 - r * r))
    response = r * balance_std + sqrt_term * noise_std
    response = RESPONSE_SLOPE_SCALER * response + RESPONSE_INTERCEPT_SCALER
    slope = r * (RESPONSE_SLOPE_SCALER / b_std)
    response_update["SLOPE"] = slope
    return response, response_update


def create_response_dataframes(noise: float, result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA) -> pd.DataFrame:
    """Creates a DataFrame containing the generated response values.

    :param noise: The standard deviation of the Gaussian noise to be added.
    :param result_dir: Directory path where the results will be saved.
    :return: A Pandas DataFrame containing the generated response values.
    """

    for lt_method in ["pairs", "CLR"]:

        taxa_info = pd.DataFrame()

        response_data_file_name_path = set_up_response_data_path(noise, lt_method, result_dir)
        response_df = pd.DataFrame()
        correlation_types = ["correlated", "uncorrelated"]
        for correlation in correlation_types:

            for response_idx in range(NUM_OF_RESPONSES_BY_TYPE):
                response_name = f"{correlation}_response_{response_idx + 1}"
                response_series, upd = create_response_series(noise, response_name, result_dir, correlation, lt_method)
                response_df = pd.concat([response_df, response_series], axis=1)
                if correlation == "correlated":
                    taxa_info = pd.concat([taxa_info, pd.DataFrame([upd])], ignore_index=True)

        response_df.to_csv(response_data_file_name_path, sep="\t")

        data_file_name, data_file_extension = os.path.splitext(response_data_file_name_path)
        taxa_info_file_path = f"{data_file_name}_taxa_info{data_file_extension}"
        taxa_info.to_csv(taxa_info_file_path, sep="\t", index=False)

    return response_df


def simulate_response_data(result_dir: str | Path = DEFAULT_PATH_TO_SIMULATED_DATA) -> None:
    """Simulates response data for all specified noise levels and saves them to files.

    :param result_dir: Directory path where the results will be saved.
    """
    for noise_level in NOISE_LEVEL_LIST:
        create_response_dataframes(noise_level, result_dir)

if __name__ == "__main__":
    simulate_microbiome_data()
    simulate_response_data()
