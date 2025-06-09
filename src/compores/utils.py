import os
import pickle
from os import PathLike
from typing import Union, AnyStr

import numpy as np
import pandas as pd
import yaml
from scipy.stats import genextreme


def load_configuration(cfg_file_path: str) -> dict:
    """This function loads the configuration file"""
    with open(cfg_file_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def save_file(dictionary: any, file_name: str, path_to_save: str | PathLike) -> None:
    """
    This function saves a dictionary to a pickle file
    :param dictionary: The dictionary to save
    :param file_name: The name of the file
    :param path_to_save: The path to save the file
    :return: None
    """
    try:
        # Ensure the target directory exists
        isExist = os.path.exists(path_to_save)
        if not isExist:
            os.makedirs(path_to_save)

        with open(os.path.join(path_to_save, file_name), 'wb') as file_handler:
            pickle.dump(dictionary, file_handler)
    except OSError:
        raise


def load_file(file_name: str, path_to_load_from: AnyStr) -> any:
    with open(os.path.join(path_to_load_from, file_name), 'rb') as file:
        return pickle.load(file)


def invert_dict(input_dict: dict) -> dict:
    inverted_dict = {}
    for key, values in input_dict.items():
        for value in values:
            if value not in inverted_dict:
                inverted_dict[value] = []
            inverted_dict[value].append(key)
    return inverted_dict


def cast_nested_dict_to_array(input_dict: dict) -> dict:
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = cast_dict_to_array(value)
    return output_dict


def cast_dict_to_array(input_dict: dict) -> np.ndarray:
    max_index = max(input_dict.keys())
    output_array = np.zeros(max_index + 1)
    for key, value in input_dict.items():
        output_array[key] = value
    return output_array


def extend_instance(
        current_instance: dict | np.ndarray, instance_to_merge: dict | np.ndarray) -> dict | np.ndarray | None:
    if isinstance(instance_to_merge, np.ndarray) and isinstance(current_instance, np.ndarray):
        return np.append(current_instance, instance_to_merge)
    elif isinstance(instance_to_merge, dict) and isinstance(current_instance, dict):
        for k, v in instance_to_merge.items():
            current_instance[k] = v
        return current_instance
    else:
        raise ValueError("No fit between instance types: at least on of the instances is of a wrong type.")


def bootstrap_p_value(observed_value, shuffled_values):
    """Calculates bootstrapped p-value with continuity correction for one-tailed test"""
    non_nan_shuffled_values = shuffled_values[~np.isnan(shuffled_values)]
    p_value = (np.sum(non_nan_shuffled_values >= observed_value) + 1) / (len(non_nan_shuffled_values) + 1)
    return p_value


def shuffle_samples(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Shuffles samples by shuffling sample codes and assigning new code to every value (row). Returns sorted data.
    """
    shuffled_indices = np.random.permutation(data.index)
    shuffled_data = data.copy()
    shuffled_data.index = shuffled_indices
    shuffled_data.sort_index(inplace=True)

    return shuffled_data


def shuffle_sample_values(data: pd.DataFrame) -> pd.DataFrame:
    """Shuffles the values for every sample (row)."""

    shuffled_data = data.copy().values

    def shuffle_and_return(arr):
        np.random.shuffle(arr)
        return arr

    np.apply_along_axis(shuffle_and_return, axis=1, arr=shuffled_data)
    shuffled_data = pd.DataFrame(shuffled_data, index=data.index, columns=data.columns)
    return shuffled_data


def calculate_root_mean_square_error(x, y, slope, intercept):
    """Calculates the noise based on the balance, response, and linear constant.

    :param x: Array-like, an independent variable array.
    :param y: Array-like, a dependent variable array.
    :param slope: Float, the slope of the linear regression.
    :param intercept: Float, the intercept constant of a linear regression.
    :return: Float, the root-mean-square noise value.
    """
    x_np = np.array(x)
    y_np = np.array(y)

    residuals = y_np - slope * x_np - intercept
    return np.sqrt(np.mean(np.square(residuals)))


def fetch_synthetic_analysis_input_data(
        path_to_outputs: str, microbiome_file_name: str, balance_method: str, response_tag: str | None
):
    """Fetches parameters for synthetic data generation: RMSE and fitted model parameters for given response values.
    :param path_to_outputs: The path to CompoRes outputs.
    :param microbiome_file_name: The name of the microbiome file.
    :param balance_method: The compositional data analysis balance method used.
    :param response_tag: The response tag for the response to apply synthetic power analysis to.
    """

    # Load the source data from the pickle files
    compores_basic_output_path = os.path.join(
        path_to_outputs, 'compores_basic_results', microbiome_file_name, balance_method
    )
    rmse_data = load_file('rmse.pkl', compores_basic_output_path)
    slope_data = load_file('slope.pkl', compores_basic_output_path)
    intercept_data = load_file('intercept.pkl', compores_basic_output_path)
    response_labels = load_file('response_index.pkl', compores_basic_output_path)
    ocu_dictionary = load_file('ocu_dictionary_compores_enriched.pkl', compores_basic_output_path)

    # Extract the array values
    rmse_values = rmse_data[microbiome_file_name]
    slope_values = slope_data[microbiome_file_name]
    intercept_values = intercept_data[microbiome_file_name]

    full_otu_list = []
    if balance_method == 'CLR':
        otu_number = max(rmse_values.keys())
        for key in ocu_dictionary[f'{otu_number} OCUs']['OCUs'].keys():
            full_otu_list.append(ocu_dictionary[f'{otu_number} OCUs']['OCUs'][key]['taxa'][0])

    response_data = {
        "rmse": [],
        "slope": [],
        "intercept": [],
        "num_ocu": [],
        "den_ocu": [],
        "ocu_number": [],
    }

    i = response_labels.index(response_tag)
    for ocu_key in rmse_values:
        response_data["rmse"].append(rmse_values[ocu_key][i])
        response_data["slope"].append(slope_values[ocu_key][i])
        response_data["intercept"].append(intercept_values[ocu_key][i])
        response_data["num_ocu"].append(ocu_dictionary[f"{ocu_key} OCUs"]['NUM_OCU'][response_tag])
        if balance_method == 'pairs':
            response_data["den_ocu"].append(ocu_dictionary[f"{ocu_key} OCUs"]['DEN_OCU'][response_tag])
        elif balance_method == 'CLR':
            response_data["den_ocu"].append(full_otu_list)
        response_data["ocu_number"].append(ocu_key)

    return response_data, response_tag


def deduplicate_synthetic_analysis_input_data(input_data_dict: dict) -> dict:

    rmse_values = input_data_dict["rmse"]
    slopes = input_data_dict["slope"]
    intercepts = input_data_dict["intercept"]
    num_ocu_s = input_data_dict["num_ocu"]
    den_ocu_s = input_data_dict["den_ocu"]
    ocu_numbers = input_data_dict["ocu_number"]

    rmse_values = np.array(round_value_list_to_significant_digits(rmse_values))
    unique_indices = np.unique(rmse_values, return_index=True)[1]
    rmse_values = rmse_values[unique_indices]
    slopes = np.array(round_value_list_to_significant_digits(slopes))[unique_indices]
    intercepts = np.array(round_value_list_to_significant_digits(intercepts))[unique_indices]
    num_ocu_s = [item for item in num_ocu_s if num_ocu_s.index(item) in unique_indices]
    den_ocu_s = [item for item in den_ocu_s if den_ocu_s.index(item) in unique_indices]
    ocu_numbers = [item for item in ocu_numbers if ocu_numbers.index(item) in unique_indices]

    input_data_dict["rmse"] = rmse_values
    input_data_dict["slope"] = slopes
    input_data_dict["intercept"] = intercepts
    input_data_dict["num_ocu"] = num_ocu_s
    input_data_dict["den_ocu"] = den_ocu_s
    input_data_dict["ocu_number"] = ocu_numbers

    return input_data_dict


def is_considered_imputed_sample(
        sample_name: str,
        num: str,
        den: str,
        ocu_dict: dict[str, dict[str, float | dict[str, dict[str, str]]]],
        total_ocu_number: int
) -> bool:
    """True if the sample is considered imputed, False otherwise.

    :param sample_name: The tag name of the given sample.
    :param num: OCU key for the OCU in the LR transformation numerator.
    :param den: OCU key for the OCU in the LR transformation denominator.
    :param ocu_dict: OCU metadata dictionary, stored in `preprocessing_results/microbiome`.
    :param total_ocu_number: The number of OCUs at the given stage of clustering.
    :return: Returns true if 50% and more of the taxa in LR numerator and denominator together were imputed.
    """
    num_imputed_otus = ocu_dict[f"{total_ocu_number} OCUs"]['OCUs'][num]
    imputed_num = len(num_imputed_otus['taxa']) if (sample_name in num_imputed_otus['imputed_in']) else 0
    if den != 'CLR':
        den_imputed_otus = ocu_dict[f"{total_ocu_number} OCUs"]['OCUs'][den]
        imputed_den = len(den_imputed_otus['taxa']) if (sample_name in den_imputed_otus['imputed_in']) else 0
    else:
        imputed_den = 0
        den_imputed_otus = {'taxa': []}
    return True if (imputed_num + imputed_den) >= (
            0.5 * (len(num_imputed_otus['taxa']) + len(den_imputed_otus['taxa']))) else False


def gev_p_value(
        observed_value: float, shuffled_values: np.ndarray, correction_method: str | None
) -> tuple[float, tuple[float, float, float]]:
    """Estimates p-value based on estimated parameters of the GEV distribution. Includes a fallback to account for
    numerical precision issues / occasional tail behavior of observed values, which result in zero p-value: the GEV
    parameters are re-estimated with the observed value added with some weight to the shuffled cases array; that is
    done either immediately, via `p_value_correction` parameter value 'weight', or after an attempt to bootstrap the
    p-value from the estimated GEV distribution, via `p_value_correction` parameter value 'bootstrap'; alternatively
    the p-value is set to 1e-11, which is the minimum p-value that can be returned by this function.

    :param observed_value: The observation to calculate the p-value for.
    :param shuffled_values: The array of PCC values generated by shuffling cycles.
    :param correction_method: The method to use for p-value correction in case of extreme values; defaults to None.
    :return:
    """
    gev_parameters = fit_gev_distribution(shuffled_values)
    estimated_p_value = calculate_p_value_with_gev(observed_value, *gev_parameters)
    if estimated_p_value <= 1e-11:
        if correction_method == 'bootstrap':
            # Combined gev bootstrapped estimate with a fallback to weight-adjusted GEV parameters re-estimation
            estimated_p_value = tail_case_gev_bootstrapped_p_value(observed_value, *gev_parameters)
            if estimated_p_value <= 1e-11:
                # Extreme observation weight adjusted GEV parameters re-estimation
                estimated_p_value, gev_parameters = tail_case_gev_weight_adjusted_p_value(
                    observed_value, shuffled_values
                )
        elif correction_method == 'weight':
            estimated_p_value, gev_parameters = tail_case_gev_weight_adjusted_p_value(observed_value, shuffled_values)
        else:
            estimated_p_value = 1e-11
    return estimated_p_value, gev_parameters


def fit_gev_distribution(shuffled_values: np.ndarray) -> tuple[float, float, float]:
    """
    Fit the Generalized Extreme Value (GEV) distribution to the given data.

    :param shuffled_values: An array of PCC values generated by shuffling cycles
    :return: GEV parameters: shape, loc, scale
    """
    # Use only non-NaN values accumulated at the given shuffle cycle
    non_nan_shuffled_values = shuffled_values[~np.isnan(shuffled_values)]
    shape, loc, scale = genextreme.fit(np.arctanh(non_nan_shuffled_values))
    return shape, loc, scale


def calculate_p_value_with_gev(observed_value: float, shape: float, loc: float, scale: float) -> float:
    """
    The p-value is computed as the survival function (1 - CDF) of the GEV distribution.

    :param observed_value: The observation to calculate the p-value for
    :param shape: The shape parameter of the GEV distribution
    :param loc: The location parameter of the GEV distribution
    :param scale: The scale parameter of the GEV distribution
    :return: The p-value for the observation
    """
    cdf_value = genextreme.cdf(np.arctanh(observed_value), c=shape, loc=loc, scale=scale)

    return 1 - cdf_value


def tail_case_gev_bootstrapped_p_value(
        observed_value: float, shape: float, loc: float, scale: float, n_samples: int = 100000000
) -> float:
    """Generates synthetic samples from the fitted GEV distribution and calculate the proportion of samples with
    a statistic greater than or equal to observed value to overcome numerical precision, parameter estimation,
    and tail behavior issues of estimating GEV distribution parameters.
    :param observed_value: The observation to calculate the p-value for;
    :param shape: The shape parameter of the GEV distribution;
    :param loc: The location parameter of the GEV distribution;
    :param scale: The scale parameter of the GEV distribution;
    :param n_samples: The number of bootstrap samples to generate;
    :return: The p-value for the observation.
    """

    # sample from the fitted GEV distribution
    synthetic_samples = genextreme.rvs(c=shape, loc=loc, scale=scale, size=n_samples)
    return np.sum(synthetic_samples >= np.arctanh(observed_value)) / n_samples


def tail_case_gev_weight_adjusted_p_value(
        observed_value: float, basic_array_for_gev_estimate: np.ndarray, weight_factor: float = .99
) -> tuple[float, tuple[float, float, float]]:
    """Calculates the p-value using the GEV distribution with epsilon weight adjustment for an extreme observed value

    :param observed_value: The observation to calculate the p-value for
    :param basic_array_for_gev_estimate: The array of values used to fit the GEV distribution before the extreme value
    :param weight_factor: The weight to apply to the extreme value
    :return: The adjusted p-value for the observation
    """
    # Ref-fit the GEV params with the observed value included with a small weight to account for the extreme magnitude
    gev_parameters = fit_gev_distribution(np.append(basic_array_for_gev_estimate, weight_factor*observed_value))
    return calculate_p_value_with_gev(observed_value, *gev_parameters), gev_parameters


def round_value_to_significant_digits(value: float, sig_digits: int = 3) -> float:
    """
    Round a value to a specified number of significant digits.

    :param value: The value to round.
    :param sig_digits: The number of significant digits to round to.
    :return: The rounded value.
    """
    if value == 0:
        return 0
    else:
        return round(value, sig_digits - int(np.floor(np.log10(abs(value)))) - 1)


def round_value_list_to_significant_digits(value_list: list[float], sig_digits: int = 3) -> list[float]:
    """
    Round a list of values to a specified number of significant digits.

    :param value_list: The list of values to round.
    :param sig_digits: The number of significant digits to round to.
    :return: The list of rounded values.
    """
    return [round_value_to_significant_digits(value, sig_digits) for value in value_list]
