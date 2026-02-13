import copy
import gc
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from logging import Logger
from typing import Any, Union, AnyStr
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .compores_otu_p_value_tracing import ComporesClusteredPValueCalculations
from .compores_compute import CompoRes
from .compores_plotting import plot_ocu_best_balance_by_response, plot_correlation_signal_significance_over_ocus
from .exceptions_module import (
    DuplicatedIndices,
    MisMatchFiles,
    NonNumericDataFrameError,
    NegativeValuesDataFrameError,
    EmptyDataFrame,
    MinDataFrame,
    NoResponseLabelFound,
    OutlierCheckFailed,
)
from .logger_module import CompoResLogger
from .preprocessing import Preprocessor, PREPROCESSING_RESULTS, OUTLIER_DETECTION
from .utils import load_configuration, save_file, load_file, shuffle_samples, shuffle_sample_values, \
    calculate_root_mean_square_error, fetch_synthetic_analysis_input_data, bootstrap_p_value, extend_instance, \
    cast_nested_dict_to_array, deduplicate_synthetic_analysis_input_data, gev_p_value, fetch_full_target_response_label

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "../..", "configs", "config_synthetic_data.yaml")

# CompoRes results path constants
PLOT_PATH = 'plots'
RESPONSE_VS_BEST_BALANCE_PATH = 'response_vs_best_balance'
SIGNAL_SIGNIFICANCE_PATH = 'compores_signal_significance'

BALANCE_CALCULATION_RESULTS_PATH = 'balance_calculation_results'
SAMPLE_LEVEL_BALANCE_RESULTS = "sample_level_results"
OCU_LEVEL_BALANCE_RESULTS = "ocu_level_results"
RESPONSE_RANKING = 'compores_response_ranking'
BASIC_RESULTS = 'compores_basic_results'
LOGS = 'logs'

CORRELATION_COEFFICIENT_ARRAYS_FILE_NAME = "correlation_coefficient.pkl"
OCU_DICTIONARY_FILE_NAME = "ocu_dictionary.pkl"
ENRICHED_OCU_DICTIONARY_FILE_NAME = "ocu_dictionary_compores_enriched.pkl"
SHUFFLES_ENRICHED_OCU_DICTIONARY_FILE_NAME = "ocu_dictionary_shuffles_enriched.pkl"
SAMPLED_OCU_CASES_FILE_NAME = "sampled_ocu_cases.pkl"
RESPONSE_INDEX_FILE_NAME = "response_index.pkl"
SLOPE_FILE_NAME = "slope.pkl"
RMSE_FILE_NAME = "rmse.pkl"
MEAN_RMSE_FILE_NAME = "mean_rmse.pkl"
INTERCEPT_FILE_NAME = "intercept.pkl"
CORRELATION_COEFFICIENT_SHUFFLE_ARRAYS_FILE_NAME = "pcc_shuffle_arrays.pkl"
SHUFFLE_MEDIAN_FILE_NAME = "shuffle_median.pkl"
TAXA_SHUFFLED_ARRAYS_FILE_NAME = "taxa_shuffled_arrays.pkl"
P_VALUES_FILE_NAME = "p_values.pkl"
GEV_PARAMETERS_FILE_NAME = "gev_parameters.pkl"
MEAN_LOG_P_VALUE_FILE_NAME = "minus_mean_log_p_value.pkl"
BOOTSTRAP_MEAN_LOG_P_VALUE_FILE_NAME = "bootstrap_minus_mean_log_p_value.pkl"
BOOTSTRAP_P_VALUE_FILE_NAME = "p_values_bootstrap.pkl"

CONFIDENCE_LEVELS = {"2_5": 2.5, "25": 25, "75": 75, "97_5": 97.5}

# Synthetic power analysis constants
NUMBER_OF_RESPONSES_TO_GENERATE = 30
NUMBER_OF_EXPERIMENT_REPEATS = 15
SYNTHETIC_DATA_PATH = "synthetic_data"
SYNTHETIC_ANALYSIS_RESULTS_PATH = "synthetic_power_analysis"


class OneCaseCombination:

    def __init__(
            self, logger: Logger, config: dict, plotting_flag: bool, s1: str, s2: str, s3: str, step=0, n_workers=4,
            coda_method="CLR", ocu_case=None, deduplicate: bool = False
    ):
        self.logger = logger
        self.config = config
        self.plotting_flag = plotting_flag
        self.ocu_case = ocu_case
        self.deduplicate = deduplicate
        self.s1 = str(s1)
        self.s2 = str(s2)
        self.s3 = str(s3)
        self.step = step
        self.workers_num = n_workers
        self.coda_method = coda_method
        self.corr_type = self.config["CORR"]
        self.shuffle_method = self.config["SHUFFLE"]
        self.n_shuffles = self.config["N_SHUFFLES"]
        self.shuffle_cycles = self.config["SHUFFLE_CYCLES"]
        self.ocu_sampling_rate = self.config["OCU_SAMPLING_RATE"]
        self.outputs_path = self.config["PATH_TO_OUTPUTS"]
        self.meta_data, self.ocu_clustering_results_path, self.path_to_prepared_response = self.set_paths()
        self.balance_results_path = os.path.join(
            self.outputs_path, BALANCE_CALCULATION_RESULTS_PATH, f'{s1}-{s2}-{s3}', self.coda_method
        )
        self.intermediate_results_path = os.path.join(
            self.outputs_path, BASIC_RESULTS, f'{s1}-{s2}-{s3}', self.coda_method)
        self.significance_plots_path = os.path.join(
            self.outputs_path, PLOT_PATH, SIGNAL_SIGNIFICANCE_PATH, f'{s1}-{s2}-{s3}', self.coda_method
        )
        self.response_vs_balance_plots_path = os.path.join(
            self.outputs_path, PLOT_PATH, RESPONSE_VS_BEST_BALANCE_PATH, f'{s1}-{s2}-{s3}', self.coda_method
        )
        self.log_files_path = str(os.path.join(self.outputs_path, LOGS, f'{s1}-{s2}-{s3}'))
        self.sampled_cluster_ocu_dictionary_keys = self.load_array(SAMPLED_OCU_CASES_FILE_NAME)
        self.ocu_dictionary = self.load_dict(OCU_DICTIONARY_FILE_NAME)
        self.response_index = self.load_array(RESPONSE_INDEX_FILE_NAME)
        self.correlation_coefficient_dict = self.load_dict(CORRELATION_COEFFICIENT_ARRAYS_FILE_NAME)
        self.mean_log_p_value_dict = self.load_dict(MEAN_LOG_P_VALUE_FILE_NAME)
        self.rmse_dict = self.load_dict(RMSE_FILE_NAME)
        self.slope_dict = self.load_dict(SLOPE_FILE_NAME)
        self.intercept_dict = self.load_dict(INTERCEPT_FILE_NAME)
        self.mean_rmse_dict = self.load_dict(MEAN_RMSE_FILE_NAME)
        self.state = self.load_state()
        self.ocu_limit = int(self.config["MAX_OCU"])

    def get_config(self) -> dict:
        """
        This function fetches the configuration dictionary.

        :return: The configuration dictionary
        """
        return self.config

    def get_correlation_coefficient_dict(self) -> dict:
        """
        This function fetches the resulting cluster dictionary.

        :return: The resulting cluster dictionary
        """
        return self.correlation_coefficient_dict

    def get_mean_log_p_value_dict(self) -> dict:
        """
        This function fetches the p-values dictionary.

        :return: The p-values dictionary
        """
        return self.mean_log_p_value_dict

    def get_rmse_dict(self) -> dict:
        """
        This function fetches the RMSE dictionary.

        :return: The RMSE dictionary
        """
        return self.rmse_dict

    def get_slope_dict(self) -> dict:
        """
        This function fetches the slope dictionary.

        :return: The slope dictionary
        """
        return self.slope_dict

    def get_intercept_dict(self) -> dict:
        """
        This function fetches the intercept dictionary.

        :return: The intercept dictionary
        """
        return self.intercept_dict

    def get_mean_rmse_dict(self) -> dict:
        """
        This function fetches the RMSE dictionary.

        :return: The RMSE dictionary
        """
        return self.mean_rmse_dict

    def set_paths(self) -> tuple:
        """
        This function sets the paths to the files that are used by the OneCaseCombination instance.
        """
        s1 = self.s1
        s2 = self.s2
        path_to_preprocessing_results = str(os.path.join(self.outputs_path, PREPROCESSING_RESULTS))
        return (
            os.path.join(self.config["PATH_TO_METADATA"], f"{s1}-{s2}-metadata.tsv"),
            os.path.join(self.outputs_path, PREPROCESSING_RESULTS, 'microbiome', 'OCUs'),
            os.path.join(path_to_preprocessing_results, 'response', f"{s1}-{s2}.tsv"),
        )

    def load_state(self) -> dict:
        try:
            with open(os.path.join(self.outputs_path, f'state_{self.coda_method}.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                f'{self.s1}-{self.s2}-{self.s3}': {
                    'preprocessed': False,
                    'run_comp': False,
                    'run_comp_shuffle_iter': [],
                    'significance_viz': [],
                    'otu_cumulative_p_value': False
                }
            }

    def update_state(self, key: str, value: Any):
        if f'{self.s1}-{self.s2}-{self.s3}' not in self.state:
            self.state[f'{self.s1}-{self.s2}-{self.s3}'] = {
                'preprocessed': False,
                'run_comp': False,
                'run_comp_shuffle_iter': [],
                'significance_viz': [],
                'otu_cumulative_p_value': False
            }

        if key in ['preprocessed', 'run_comp', 'otu_cumulative_p_value']:
            self.state[f'{self.s1}-{self.s2}-{self.s3}'][key] = value
        else:
            self.state[f'{self.s1}-{self.s2}-{self.s3}'][key].append(value)
        with open(os.path.join(self.outputs_path, f'state_{self.coda_method}.json'), 'w') as f:
            json.dump(self.state, f, indent=4)

    def load_array(self, array_name):
        try:
            file = load_file(array_name, self.intermediate_results_path)
            self.logger.debug(f"Loaded {os.path.join(self.intermediate_results_path, array_name)}")
            return file
        except FileNotFoundError:
            return []

    def load_dict(self, dict_name: str, subfolder_name: str = "") -> dict:
        if subfolder_name:
            path = f"{self.intermediate_results_path}/{subfolder_name}"
        else:
            path = self.intermediate_results_path
        try:
            file = load_file(dict_name, path)
            self.logger.debug(f"Loaded {os.path.join(path, dict_name)}")
            return file
        except FileNotFoundError:
            return {}

    @staticmethod
    def mean_log_score_over_otu_clustering(exp_dictionary: dict[str, dict[int, np.ndarray]]) -> dict[str, np.ndarray]:
        mean_dictionary = {}
        for key in exp_dictionary:
            mean_dictionary[key] = - np.mean(np.vstack(np.log(list(exp_dictionary[key].values()))), axis=0)
        return mean_dictionary

    @staticmethod
    def mean_score_over_otu_clustering(exp_dictionary: dict[str, dict[int, np.ndarray]]) -> dict[str, np.ndarray]:
        mean_dictionary = {}
        for key in exp_dictionary:
            mean_dictionary[key] = np.mean(np.vstack(list(exp_dictionary[key].values())), axis=0)
        return mean_dictionary

    def extract_response_tags(self) -> None:
        """Extracts response feature tags from a DataFrame and saves them to a file."""
        response = pd.read_parquet(self.path_to_prepared_response)
        self.response_index = [
            f'response_{i + 1}_{col}' for i, col in enumerate(response.columns)
        ]
        self.response_index = [col.split("\n")[0] for col in self.response_index]
        save_file(self.response_index, RESPONSE_INDEX_FILE_NAME, self.intermediate_results_path)

    def _read_preprocessed_results(self):

        suffix_name = f"{self.s1}-{self.s2}-{self.s3}"
        base_path = os.path.join(self.ocu_clustering_results_path, suffix_name)

        # run over the sampled ocu folders and yield the data
        for sub_folder_name in self.sampled_cluster_ocu_dictionary_keys:
            # iterate over all files in the folder
            for item in os.listdir(os.path.join(base_path, f"{sub_folder_name}")):
                if suffix_name in item and ".tsv" in item:
                    microbiome = pd.read_csv(
                        os.path.join(base_path, f"{sub_folder_name}", item), sep='\t', header=0, index_col=0
                    )
                    num_of_otu_clusters = microbiome.shape[1]
                    if num_of_otu_clusters > self.ocu_limit and self.coda_method != "CLR":
                        self.logger.warning(
                            f"Skipping {sub_folder_name} OCU case, as it exceeds the limit ({self.ocu_limit})."
                        )
                        continue

                    yield microbiome, num_of_otu_clusters, suffix_name

    def _update_clustered_ocu_dictionary(
            self, key_to_add: str, value_to_add: Any, dict_key: int, response_key: str
    ) -> None:
        self.ocu_dictionary[f"{dict_key} OCUs"][key_to_add][response_key] = value_to_add

    def _combine_and_store_ocu_level_results(self):
        """Serves the results of the CompoRes balance analysis by OCU level as CSV tables. For every response, a table
        is stored in a directory named "ocu_level_results/regular" within the balance results path on. It includes the
        following columns:
        - OCU clustering key;
        - Numerator taxa list;
        - Denominator taxa list (for SLR (pairs) CoDA method);
        - Correlation coefficient;
        - Root mean square error (RMSE) of the linera model fitted to the data;
        - Slope of the linear model;
        - Intercept of the linear model;
        - Median of the shuffled correlation coefficients;
        - Confidence intervals of the shuffled correlation coefficients;
        - p-value of the correlation coefficient bootstrapped / estimated from the shuffled data.
        """

        path_to_ocu_level_compores_result_data = os.path.join(
            self.balance_results_path,
            OCU_LEVEL_BALANCE_RESULTS,
            "regular"
        )
        os.makedirs(path_to_ocu_level_compores_result_data, exist_ok=True)

        ocu_dictionary_compores_enriched = self.load_dict(ENRICHED_OCU_DICTIONARY_FILE_NAME)
        ocu_dictionary_shuffle_enriched = self.load_dict(SHUFFLES_ENRICHED_OCU_DICTIONARY_FILE_NAME)
        ocu_dictionary_shuffle_enriched = {
            f"{k} OCUs": v for k, v in ocu_dictionary_shuffle_enriched.items()
        }

        highest_correlation_taxa_dict = {}
        for key in ocu_dictionary_compores_enriched:
            self.logger.debug(f"Combining results for {key}.")
            for item in ocu_dictionary_compores_enriched[key]["NUM_OCU"].items():
                if item[0] not in highest_correlation_taxa_dict:
                    highest_correlation_taxa_dict[item[0]] = {}
                highest_correlation_taxa_dict[item[0]][key] = {}
                highest_correlation_taxa_dict[item[0]][key]["NUM_Taxa_List"] = item[1]
            if ocu_dictionary_compores_enriched[key]["DEN_OCU"]:
                for item in ocu_dictionary_compores_enriched[key]["DEN_OCU"].items():
                    highest_correlation_taxa_dict[item[0]][key]["DEN_Taxa_List"] = item[1]

            compores_result_keys = ["rho", "rmse", "slope", "intercept"]
            for result_key in compores_result_keys:
                for item in ocu_dictionary_compores_enriched[key][result_key].items():
                    highest_correlation_taxa_dict[item[0]][key][result_key] = item[1]

            compores_shuffle_result_keys = ["rho_shuffle_median", "rho_p_value_bootstrap", "rho_p_value"]
            for level in CONFIDENCE_LEVELS:
                compores_shuffle_result_keys.append(f"rho_shuffle_ci_{level}")
            for result_key in compores_shuffle_result_keys:
                for item in ocu_dictionary_shuffle_enriched[key][result_key].items():
                    highest_correlation_taxa_dict[item[0]][key][result_key] = item[1]

                    with open(
                            os.path.join(path_to_ocu_level_compores_result_data,
                                         f"{item[0]}_combined_ocu_level_results.json"
                                         ), 'w'
                    ) as f:
                        json.dump(highest_correlation_taxa_dict[item[0]], f, indent=4)

        for key in highest_correlation_taxa_dict:

            ocu_clustering_key_list = []
            num_taxa_list = []
            den_taxa_list = []
            rho_list = []
            rmse_list = []
            slope_list = []
            intercept_list = []
            rho_shuffle_median_list = []
            rho_shuffle_ci_dict = {}
            for level in CONFIDENCE_LEVELS:
                rho_shuffle_ci_dict[level] = []
            rho_p_value_bootstrap_list = []
            rho_p_value_list = []

            for item in highest_correlation_taxa_dict[key].items():
                ocu_clustering_key_list.append(item[0])
                num_taxa_list.append(item[1]["NUM_Taxa_List"])
                if "DEN_Taxa_List" in item[1]:
                    den_taxa_list.append(item[1]["DEN_Taxa_List"])
                rho_list.append(item[1]["rho"])
                rmse_list.append(item[1]["rmse"])
                slope_list.append(item[1]["slope"])
                intercept_list.append(item[1]["intercept"])
                rho_shuffle_median_list.append(item[1]["rho_shuffle_median"])
                for level in CONFIDENCE_LEVELS:
                    rho_shuffle_ci_dict[level].append(item[1][f"rho_shuffle_ci_{level}"])
                rho_p_value_bootstrap_list.append(item[1]["rho_p_value_bootstrap"])
                rho_p_value_list.append(item[1]["rho_p_value"])

            response_taxa_dict = {"NUM_Taxa_List": num_taxa_list}
            if den_taxa_list:
                response_taxa_dict["DEN_Taxa_List"] = den_taxa_list
            response_taxa_dict["rho"] = rho_list
            response_taxa_dict["rmse"] = rmse_list
            response_taxa_dict["slope"] = slope_list
            response_taxa_dict["intercept"] = intercept_list
            response_taxa_dict["rho_shuffle_median"] = rho_shuffle_median_list
            for level in CONFIDENCE_LEVELS:
                response_taxa_dict[f"rho_shuffle_ci_{level}"] = rho_shuffle_ci_dict[level]
            response_taxa_dict["rho_p_value_bootstrap"] = rho_p_value_bootstrap_list
            response_taxa_dict["rho_p_value"] = rho_p_value_list

            taxa_df = pd.DataFrame(response_taxa_dict, index=ocu_clustering_key_list)
            with open(os.path.join(
                    path_to_ocu_level_compores_result_data, f"{key}_combined_ocu_level_results.csv"), 'w'
            ) as f:
                taxa_df.to_csv(f, index=True)

    def _combine_and_store_ocu_level_shuffle_results(self):
        """Serves the results of running CompoRes analysis on shuffled data; the numerator and denominator (for the
        SLR (pairs) CoDA method) taxa lists are stored as JSON files with nested arrays of numerator / denominator
        taxa for every shuffling with OCU labels as keys. For every response, the result is stored in a directory named
        "ocu_level_results/shuffled" within the balance results path.
        """

        path_to_ocu_level_compores_shuffle_result_data = os.path.join(
            self.balance_results_path,
            OCU_LEVEL_BALANCE_RESULTS,
            "shuffled"
        )
        os.makedirs(path_to_ocu_level_compores_shuffle_result_data, exist_ok=True)

        taxa_shuffled_arrays = self.load_dict(TAXA_SHUFFLED_ARRAYS_FILE_NAME)

        for exp_name in taxa_shuffled_arrays:
            for ocu_num in taxa_shuffled_arrays[exp_name]:
                ocu_key = f"{ocu_num} OCUs"
                for item in taxa_shuffled_arrays[exp_name][ocu_num].items():
                    response_tag = self.response_index[item[0]]
                    result_data = {ocu_key: {"NUM_Taxa_Lists": item[1][0]}}
                    if self.coda_method != "CLR":
                        result_data[ocu_key]["DEN_Taxa_Lists"] = item[1][1]
                    json_file_path = os.path.join(
                        path_to_ocu_level_compores_shuffle_result_data,
                        f"{response_tag}_combined_ocu_level_shuffling_results.json"
                    )
                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r+') as f:
                            existing_data = json.load(f)
                            existing_data.update(result_data)
                            f.seek(0)
                            json.dump(existing_data, f, indent=4)
                    else:
                        with open(json_file_path, 'w') as f:
                            json.dump(result_data, f, indent=4)

        taxa_shuffled_arrays.clear()

    def run_comp_process_response_subtask(
        self,
        response_index: int,
        response_tag_str: str,
        microbiome_df: pd.DataFrame,
        response_df: pd.DataFrame,
        exp_name_str: str,
        total_ocu: int,
        outlier_detection_step_flag: bool = False,
        outlier_detection_all: bool = False,
        outlier_detection_masked: bool = False,
        max_ind: str = None,
    ):
        """The function runs a CompoRes core task for a specific response feature."""
        value = response_df.iloc[:, response_index].copy()
        response_name = value.name.split('\n')[0]
        if not outlier_detection_step_flag:
            log_message = f"Processing {exp_name_str}.response_{response_index + 1}_{response_name}.{total_ocu}_OCUs"
            self.logger.info(log_message)
        compores_object = CompoRes(
            microbiome_df,
            value,
            total_ocu,
            self.log_files_path,
            self.intermediate_results_path,
            balance_method=self.coda_method,
            corr_type=self.corr_type,
            regular_run_flag=self.plotting_flag
        )
        compores_object.run_analysis(outlier_detection_step_flag=outlier_detection_step_flag, max_ind=max_ind)
        result = compores_object.get_results()

        if self.plotting_flag and not outlier_detection_step_flag:

            response_string = f"response_{response_index + 1}_{response_name}"
            path_to_sample_level_compores_result_data = os.path.join(
                self.balance_results_path,
                SAMPLE_LEVEL_BALANCE_RESULTS,
                response_string
            )
            os.makedirs(path_to_sample_level_compores_result_data, exist_ok=True)

            output_name = f"{exp_name_str}-ocu_{total_ocu}-{response_string}"
            with open(os.path.join(path_to_sample_level_compores_result_data, output_name + ".csv"), 'w') as f:
                result.to_csv(f, index=False)

            corr_matrix = compores_object.get_correlation_matrix()

            csv_file_path = os.path.join(
                path_to_sample_level_compores_result_data,
                f"{output_name}-clr_correlation_values.csv"
            )
            if self.coda_method == "pairs":
                csv_file_path = os.path.join(
                    path_to_sample_level_compores_result_data,
                    f"{output_name}-pairs_correlation_values.csv"
                )
            with open(csv_file_path, 'w') as f:
                corr_matrix.to_csv(f)
            del corr_matrix

        # Release memory
        del compores_object

        # TODO change to ranking for spearman
        lingress_result = stats.linregress(result['Final_LR_Value'], result['Response'])
        slope = lingress_result.slope
        intercept = lingress_result.intercept
        r_value = lingress_result.rvalue
        se = lingress_result.stderr
        # lingress_result.pvalue, lingress_result.stderr, getattr(lingress_result, 'intercept_stderr', None)
        rmse_val = calculate_root_mean_square_error(result['Final_LR_Value'], result['Response'], slope, intercept)
        num_taxa_array = result["NUM_OCU"].iloc[0]
        if self.coda_method != "CLR":
            den_taxa_array = result["DEN_OCU"].iloc[0]
        else:
            den_taxa_array = None

        if self.plotting_flag:

            response_string = f"{response_name}"

            if outlier_detection_step_flag:
                path_to_plots = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION, PLOT_PATH)
                if outlier_detection_all:
                    response_string = f"{response_name}_all_samples"
                elif outlier_detection_masked:
                    response_string = f"{response_name}_masked_samples"
            else:
                path_to_plots = self.response_vs_balance_plots_path

            plot_ocu_best_balance_by_response(
                self.meta_data,
                response_index,
                response_string,
                total_ocu,
                exp_name_str, result,
                path_to_plots,
                intercept,
                slope,
                r_value
            )

        return response_index, response_tag_str, r_value, rmse_val, slope, intercept, num_taxa_array, den_taxa_array, se

    def run_comp_process_task(
            self, microbiome: pd.DataFrame, total_ocu_num: int, exp_name: str, response_df: pd.DataFrame,
            outlier_detection_step_flag: bool = False,
            outlier_detection_all: bool = False, outlier_detection_masked: bool = False,
            n_taxa: dict = None
    ) -> tuple:
        """The function pools runs of the CompoRes analysis over response features in the response dataframe.

        :param microbiome: The microbiome data frame;
        :param total_ocu_num: the total number of clusters in the run;
        :param exp_name: the name of the experiment;
        :param response_df: the response data frame;
        :param outlier_detection_step_flag: whether the current run is for outlier detection;
        :param outlier_detection_all: whether the outlier detection run is on all data;
        :param outlier_detection_masked:  whether the outlier detection run is on data without candidate sample points;
        :param n_taxa: optional, the dictionary of the nominator taxa per response feature;
        :return: tuple of calculated values: pcc list, rmse list, and a dictionary of response level information.
        """
        response = response_df.copy()
        clustered_ocu_dictionary_enrichment = {f"{total_ocu_num} OCUs": {}}
        for result_key in ["NUM_OCU", "DEN_OCU", "rho", "rmse", "slope", "intercept"]:
            clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"][result_key] = {}
        if not outlier_detection_step_flag:
            self.logger.info(f"Starting processing {exp_name} with {total_ocu_num} OCUs.")

        # TODO: add feature-index json
        num_of_features = response.shape[1]
        pcc = np.zeros(num_of_features)
        rmse = np.zeros(num_of_features)
        sl = np.zeros(num_of_features)
        ic = np.zeros(num_of_features)
        se = np.zeros(num_of_features)

        try:
            with ProcessPoolExecutor(max_workers=self.workers_num) as res_executor:
                res_futures = []
                self.logger.debug(f"Spawning {len(self.response_index)} tasks; res_futures is {res_futures}")
                self.logger.debug(f"Response index: {self.response_index}")
                for res_i, response_tag in enumerate(self.response_index):
                    self.logger.debug(
                        f"Starting processing {response_tag}."
                    )
                    res_futures.append(
                        res_executor.submit(
                            self.run_comp_process_response_subtask, res_i, response_tag,
                            microbiome, response, exp_name, total_ocu_num,
                            outlier_detection_step_flag=outlier_detection_step_flag,
                            outlier_detection_all=outlier_detection_all,
                            outlier_detection_masked=outlier_detection_masked,
                            max_ind=n_taxa[response_tag] if n_taxa else None
                        )
                    )

                # Collect results from all futures
                for res_future in as_completed(res_futures):
                    (
                        res_i, res_tag, pcc_value, rmse_value, m_slope, m_intercept, n_taxa, d_taxa, m_se
                    ) = res_future.result()
                    pcc[res_i] = pcc_value
                    rmse[res_i] = rmse_value
                    sl[res_i] = m_slope
                    ic[res_i] = m_intercept
                    se[res_i] = m_se
                    clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["NUM_OCU"][res_tag] = n_taxa
                    if d_taxa:
                        clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["DEN_OCU"][res_tag] = d_taxa
                    clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["rho"][res_tag] = pcc_value
                    clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["rmse"][res_tag] = rmse_value
                    clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["slope"][res_tag] = m_slope
                    clustered_ocu_dictionary_enrichment[f"{total_ocu_num} OCUs"]["intercept"][res_tag] = m_intercept
        except (BrokenProcessPool, OSError, ValueError):
            raise
        return total_ocu_num, pcc, rmse, sl, ic, clustered_ocu_dictionary_enrichment, se

    def run_comp_shuffles_process_response_subtask(
            self, response_feature_index: int, response_tag_str: str,
            microbiome_df: pd.DataFrame, response_value: pd.Series, total_ocu: int, start_index: int,
            pcc_shuffle_array: np.ndarray, taxa_shuffle_array_tuple: tuple[np.ndarray, np.ndarray],
            r_value, p_value_correction: str = None  # 'weight' or 'bootstrap' or None
    ):
        """Pools shuffling runs of the CompoRes core task for a specific feature-microbiome case and calculates
        two versions of correlation estimate p-value: one directly bootstrapped from the shuffling results
        and one based on estimated parameters of the GEV distribution.
        """
        exception_queue = queue.Queue()
        threads = []
        for i in range(self.n_shuffles):
            thread = threading.Thread(target=self.run_comp_shuffles_in_process_thread_task, args=(
                i + start_index, response_tag_str, response_value, microbiome_df, total_ocu,
                pcc_shuffle_array, taxa_shuffle_array_tuple, exception_queue
            ))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        if not exception_queue.empty():
            self.logger.info(f"{response_tag_str}.{total_ocu}_OCUs: Exception(s) occurred"
                             f" in the shuffle threads, check `compores_compute` logs.")

        batch_pcc_shuffle_median = np.nanmedian(pcc_shuffle_array)
        estimated_p_value, gev_parameters = gev_p_value(r_value, pcc_shuffle_array, p_value_correction)
        estimated_p_value_bootstrap = bootstrap_p_value(r_value, pcc_shuffle_array)

        return (response_feature_index, response_tag_str, pcc_shuffle_array, taxa_shuffle_array_tuple,
                batch_pcc_shuffle_median, estimated_p_value_bootstrap, estimated_p_value, gev_parameters)

    def run_comp_shuffles_process_task(
            self,
            microbiome: pd.DataFrame,
            total_ocu_num: int,
            exp_name: str,
            response_df: pd.DataFrame,
            shuffling_cycle: int,
            batch_indices: list,
            r_values_array: np.ndarray,
            pcc_shuffle_arrays: dict,
            taxa_shuffle_array_tuples: dict
    ):
        """The function pools CompoRes analysis runs with shuffling over all target features for a specific microbiome
        and combines the shuffling results in two main dictionaries: for overall results and for enriching the OCU
        clustering information file.
        """

        first_nan_i = (shuffling_cycle - 1) * self.n_shuffles

        clustered_ocu_dictionary_enrichment = {}
        basic_shuffle_result = {
            "pcc_shuffle_arrays": {},
            "taxa_shuffle_array_tuples": {},
            "gev_parameters": {},
            "pcc_shuffle_median": {},
            "p_values_bootstrap": {},
            "p_values": {}
        }
        for level in CONFIDENCE_LEVELS:
            basic_shuffle_result[f"rho_shuffle_ci_{level}"] = {}

        for level in CONFIDENCE_LEVELS:
            if f"rho_shuffle_ci_{level}" not in clustered_ocu_dictionary_enrichment:
                clustered_ocu_dictionary_enrichment[f"rho_shuffle_ci_{level}"] = {}

        for result_key in ["rho_shuffle_median", "rho_p_value_bootstrap", "rho_p_value"]:
            if result_key not in clustered_ocu_dictionary_enrichment:
                clustered_ocu_dictionary_enrichment[result_key] = {}
        try:
            with ProcessPoolExecutor(max_workers=self.workers_num) as res_executor:
                res_futures = []
                for enum_index, response_tag in enumerate([self.response_index[i] for i in batch_indices]):
                    res_features_processed = batch_indices[enum_index]
                    value = response_df.iloc[:, enum_index].copy()
                    if self.plotting_flag:
                        self.logger.info(
                            f"Shuffling: {total_ocu_num}_OCUs.{response_tag}"
                        )

                    r_value = r_values_array[res_features_processed]

                    if res_features_processed in pcc_shuffle_arrays:
                        pcc_shuffle_array = pcc_shuffle_arrays[res_features_processed]
                        taxa_shuffle_array_tuple = taxa_shuffle_array_tuples[res_features_processed]
                    else:
                        pcc_shuffle_array = np.full(self.n_shuffles * self.shuffle_cycles, np.nan)
                        taxa_shuffle_array_tuple = (
                            [None] * self.n_shuffles * self.shuffle_cycles,
                            [None] * self.n_shuffles * self.shuffle_cycles
                        )
                    if self.plotting_flag:
                        self.logger.info(
                            f"non-NaN values: {np.sum(~np.isnan(pcc_shuffle_array))}; update from index {first_nan_i}"
                        )

                    res_futures.append(
                        res_executor.submit(
                            self.run_comp_shuffles_process_response_subtask, res_features_processed, response_tag,
                            microbiome, value, total_ocu_num, first_nan_i,
                            pcc_shuffle_array, taxa_shuffle_array_tuple, r_value
                        )
                    )

                # Collect results from all futures
                for res_future in as_completed(res_futures):
                    (res_features_processed, res_tag, pcc_array, taxa_array_tuple,
                     pcc_shuffle_median, p_value_bootstrap, p_value, gev_parameters) = res_future.result()

                    basic_shuffle_result["pcc_shuffle_arrays"][res_features_processed] = pcc_array
                    basic_shuffle_result["taxa_shuffle_array_tuples"][res_features_processed] = taxa_array_tuple
                    basic_shuffle_result["pcc_shuffle_median"][res_features_processed] = pcc_shuffle_median
                    basic_shuffle_result["p_values_bootstrap"][res_features_processed] = p_value_bootstrap
                    basic_shuffle_result["p_values"][res_features_processed] = p_value
                    basic_shuffle_result["gev_parameters"][res_features_processed] = dict(
                        zip(["shape", "loc", "scale"], gev_parameters)
                    )
                    for level, value in CONFIDENCE_LEVELS.items():
                        basic_shuffle_result[
                            f"rho_shuffle_ci_{level}"
                        ][res_features_processed] = np.nanpercentile(pcc_array, value)

                    clustered_ocu_dictionary_enrichment["rho_shuffle_median"][res_tag] = pcc_shuffle_median
                    clustered_ocu_dictionary_enrichment["rho_p_value_bootstrap"][res_tag] = p_value_bootstrap
                    clustered_ocu_dictionary_enrichment["rho_p_value"][res_tag] = p_value
                    for level, value in CONFIDENCE_LEVELS.items():
                        clustered_ocu_dictionary_enrichment[
                            f"rho_shuffle_ci_{level}"
                        ][res_tag] = np.nanpercentile(pcc_array, value)
        except (BrokenProcessPool, OSError, ValueError):
            raise

        return exp_name, total_ocu_num, basic_shuffle_result, clustered_ocu_dictionary_enrichment, batch_indices

    def run_comp_shuffles_in_process_thread_task(
            self, thread_id: int, value_tag_str: str,
            value: pd.Series, microbiome: pd.DataFrame, ocu_num: int,
            pcc_shuffle_array: np.ndarray, taxa_shuffle_array_tuple: tuple[np.ndarray, np.ndarray],
            exception_queue: queue.Queue
    ):
        """Shuffles the values of the response or microbiome data and runs analysis the predefined number of times.
        As a side effect updates the `pcc_shuffle` array of the pearson correlation coefficients resulting from shuffled
        data.

        :param value_tag_str: The name tag of the response feature; includes its index;
        :param value: response series;
        :param microbiome: microbiome dataframe;
        :param ocu_num: number of OCU clusters in the run;
        :param thread_id: thread counter;
        :param pcc_shuffle_array: array of pearson correlation coefficients resulting from shuffled data;
        :param taxa_shuffle_array_tuple: tuple taxa list arrays resulting from shuffled data;
        :param exception_queue: queue for catching exceptions in threads;
        :return:
        """
        shuffled_values = value.copy()
        shuffled_microbiome = microbiome.copy()
        if self.shuffle_method == "response":
            shuffled_values = shuffle_samples(shuffled_values)
        elif self.shuffle_method == "microbiome":
            shuffled_microbiome = shuffle_samples(shuffled_microbiome)
            shuffled_microbiome = shuffle_sample_values(shuffled_microbiome)
        compores_object = CompoRes(
            shuffled_microbiome,
            shuffled_values,
            ocu_num,
            self.log_files_path,
            self.intermediate_results_path,
            balance_method=self.coda_method,
            corr_type=self.corr_type,
            shuffle_flag=True,
            regular_run_flag=self.plotting_flag
        )
        try:
            compores_object.run_analysis()
        except Exception as e:
            compores_object.logger.error(f"Error in thread {thread_id} for {value_tag_str}: {e}")
            exception_queue.put(e)
        finally:
            res = compores_object.get_results()
            del compores_object, shuffled_values, shuffled_microbiome
            slope, intercept, r_value, p_value, std_err = stats.linregress(res['Final_LR_Value'], res['Response'])

            pcc_shuffle_array[thread_id] = abs(r_value)
            # fetch taxa mapping for the shuffled result
            num_taxa_list = res["NUM_OCU"].iloc[0]
            taxa_shuffle_array_tuple[0][thread_id] = num_taxa_list
            if self.coda_method != "CLR":
                den_taxa_list = res["DEN_OCU"].iloc[0]
                taxa_shuffle_array_tuple[1][thread_id] = den_taxa_list

    def pool_run_compores_over_preprocessed_case(self) -> None:
        """
        This function runs the CompoRes analysis over sampled preprocessed OCU files and updates a resulting
        dictionary with correlation coefficient value for a corresponding OTU clustering case and response, in
        accordance with the CODA_METHOD and CORR type chosen.

        :return: None
        """
        response = pd.read_parquet(self.path_to_prepared_response)
        temp_dict = {}
        try:
            with ProcessPoolExecutor(max_workers=self.workers_num) as executor:
                futures = []
                for microbiome_data, num_of_ocu_s, suffix_name in self._read_preprocessed_results():
                    futures.append(executor.submit(
                        self.run_comp_process_task, microbiome_data, num_of_ocu_s,
                        suffix_name, response))
                    if suffix_name not in temp_dict:
                        temp_dict[suffix_name] = {}
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"] = {}
                # Wait for all futures to complete and collect results
                for future in as_completed(futures):
                    (
                        num_of_ocu_s, pcc_list, rmse_list, slope_list, intercept_list, ocu_dictionary_enrichment, _
                    ) = future.result()
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"]["pcc"] = pcc_list
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"]["rmse"] = rmse_list
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"]["slope"] = slope_list
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"]["intercept"] = intercept_list
                    temp_dict[suffix_name][f"{num_of_ocu_s} OCUs"]["ocu_dictionary"] = (
                        ocu_dictionary_enrichment[f"{num_of_ocu_s} OCUs"]
                    )
        except (BrokenProcessPool, OSError, ValueError):
            raise

        # Update the instance dictionaries after all processes are complete
        ocu_dictionary = load_file('ocu_dictionary.pkl', self.intermediate_results_path)
        if self.coda_method != "CLR":
            # Keep only the keys that meet the max OCU number condition for CoDA methods, other than CLR
            ocu_dictionary = {
                k: v for k, v in ocu_dictionary.items() if int(k.split(' ')[0]) <= self.ocu_limit
            }
        for exp_key in temp_dict:
            if exp_key not in self.correlation_coefficient_dict:
                self.correlation_coefficient_dict[exp_key] = {}
                self.rmse_dict[exp_key] = {}
                self.slope_dict[exp_key] = {}
                self.intercept_dict[exp_key] = {}
            for ocu_key in temp_dict[exp_key]:
                dir_name = int(ocu_key.split(' ')[0])
                self.correlation_coefficient_dict[exp_key][dir_name] = temp_dict[exp_key][ocu_key]["pcc"]
                self.rmse_dict[exp_key][dir_name] = temp_dict[exp_key][ocu_key]["rmse"]
                self.slope_dict[exp_key][dir_name] = temp_dict[exp_key][ocu_key]["slope"]
                self.intercept_dict[exp_key][dir_name] = temp_dict[exp_key][ocu_key]["intercept"]
                for result_key in temp_dict[exp_key][ocu_key]["ocu_dictionary"]:
                    ocu_dictionary[ocu_key][result_key] = (
                        temp_dict[exp_key][ocu_key]["ocu_dictionary"][result_key]
                    )

        # Release memory
        temp_dict.clear()
        del temp_dict

        self.mean_rmse_dict = OneCaseCombination.mean_score_over_otu_clustering(self.rmse_dict)

        # Save dictionaries as intermediate results to pickle files
        save_file(
            self.correlation_coefficient_dict, CORRELATION_COEFFICIENT_ARRAYS_FILE_NAME, self.intermediate_results_path
        )
        save_file(self.rmse_dict, RMSE_FILE_NAME, self.intermediate_results_path)
        save_file(self.slope_dict, SLOPE_FILE_NAME, self.intermediate_results_path)
        save_file(self.intercept_dict, INTERCEPT_FILE_NAME, self.intermediate_results_path)
        save_file(self.mean_rmse_dict, MEAN_RMSE_FILE_NAME, self.intermediate_results_path)

        # Convert the OCU dictionary entries to taxa lists
        for ocu_key in ocu_dictionary:
            for result_key in ["NUM_OCU", "DEN_OCU"]:
                if result_key in ocu_dictionary[ocu_key]:
                    for response_key in ocu_dictionary[ocu_key][result_key]:
                        ocu_dictionary[ocu_key][result_key][response_key] = ocu_dictionary[ocu_key]['OCUs'][
                            ocu_dictionary[ocu_key][result_key][response_key]
                        ]['taxa']

        # Save the OCU dictionary with enriched taxa lists and clear it from memory
        save_file(ocu_dictionary, ENRICHED_OCU_DICTIONARY_FILE_NAME, self.intermediate_results_path)
        ocu_dictionary.clear()
        del ocu_dictionary

    def pool_run_compores_shuffles_over_preprocessed_case(
            self, shuffling_cycle: int, batch_size: int = 100
    ) -> None:
        """Runs the CompoRes analysis over sampled OTU clustering cases, while performing the
        predefined number of shuffles, and updates the resulting dictionaries: shuffle median, shuffle CIs, and
        p-values for Pearson Correlation Coefficient value for a corresponding OTU clustering case and response, in
        accordance with the CODA_METHOD and CORR type chosen. It also stores Pearson Correlation Coefficient arrays
        and taxa lists for each shuffle. The intermediate results are stored in batches of size `batch_size` in the
        intermediate results directory; combined dictionaries are constructed at the end of the function.

        :return: None
        """

        response = pd.read_parquet(self.path_to_prepared_response)
        num_of_features = response.shape[1]
        # Split response indices into batches
        response_indices = list(range(num_of_features))
        batches = [response_indices[i:i + batch_size] for i in range(0, num_of_features, batch_size)]

        for batch in batches:
            self.logger.info(f"Processing batch {batch}")
            response_slice = response.iloc[:, batch]

            try:
                with ProcessPoolExecutor(max_workers=self.workers_num) as executor:
                    futures = []
                    for microbiome_data, num_of_ocu_s, suffix_name in self._read_preprocessed_results():

                        os.makedirs(os.path.join(
                            self.intermediate_results_path, f"{batch[0]}-{batch[-1]}"), exist_ok=True
                        )
                        pcc_shuffle_arrays = self.load_dict(CORRELATION_COEFFICIENT_SHUFFLE_ARRAYS_FILE_NAME,
                                                            f"{batch[0]}-{batch[-1]}")
                        taxa_shuffled_arrays = self.load_dict(TAXA_SHUFFLED_ARRAYS_FILE_NAME,
                                                              f"{batch[0]}-{batch[-1]}")
                        shuffle_median_dict = self.load_dict(SHUFFLE_MEDIAN_FILE_NAME,
                                                             f"{batch[0]}-{batch[-1]}")
                        p_values_bootstrap_dict = self.load_dict(BOOTSTRAP_P_VALUE_FILE_NAME,
                                                                 f"{batch[0]}-{batch[-1]}")
                        p_values_dict = self.load_dict(P_VALUES_FILE_NAME,
                                                       f"{batch[0]}-{batch[-1]}")
                        gev_parameters_dict = self.load_dict(GEV_PARAMETERS_FILE_NAME,
                                                             f"{batch[0]}-{batch[-1]}")
                        shuffle_ci_dict = {}
                        for level in CONFIDENCE_LEVELS:
                            shuffle_ci_dict[level] = self.load_dict(
                                f"shuffle_ci_{level}.pkl", f"{batch[0]}-{batch[-1]}"
                            )
                        ocu_dictionary_shuffle_enrichment = self.load_dict(
                            OCU_DICTIONARY_FILE_NAME, f"{batch[0]}-{batch[-1]}"
                        )
                        if suffix_name not in pcc_shuffle_arrays:
                            pcc_shuffle_arrays[suffix_name] = {}
                            taxa_shuffled_arrays[suffix_name] = {}
                            shuffle_median_dict[suffix_name] = {}
                            p_values_bootstrap_dict[suffix_name] = {}
                            p_values_dict[suffix_name] = {}
                            gev_parameters_dict[suffix_name] = {}
                            for level in CONFIDENCE_LEVELS:
                                shuffle_ci_dict[level][suffix_name] = {}

                        if num_of_ocu_s not in pcc_shuffle_arrays[suffix_name]:
                            pcc_shuffle_arrays[suffix_name][num_of_ocu_s] = {}
                            taxa_shuffled_arrays[suffix_name][num_of_ocu_s] = {}
                            shuffle_median_dict[suffix_name][num_of_ocu_s] = {}
                            p_values_bootstrap_dict[suffix_name][num_of_ocu_s] = {}
                            p_values_dict[suffix_name][num_of_ocu_s] = {}
                            gev_parameters_dict[suffix_name][num_of_ocu_s] = {}
                            for level in CONFIDENCE_LEVELS:
                                shuffle_ci_dict[level][suffix_name][num_of_ocu_s] = {}
                            ocu_dictionary_shuffle_enrichment[num_of_ocu_s] = {}

                        bound_run_comp_shuffle = partial(
                            self.run_comp_shuffles_process_task,
                            r_values_array=copy.deepcopy(self.correlation_coefficient_dict[suffix_name][num_of_ocu_s]),
                            pcc_shuffle_arrays=pcc_shuffle_arrays[suffix_name][num_of_ocu_s],
                            taxa_shuffle_array_tuples=taxa_shuffled_arrays[suffix_name][num_of_ocu_s]
                        )

                        futures.append(executor.submit(
                            bound_run_comp_shuffle, microbiome=microbiome_data,
                            total_ocu_num=num_of_ocu_s, exp_name=suffix_name, response_df=response_slice,
                            shuffling_cycle=shuffling_cycle, batch_indices=batch
                        ))

                    # Wait for all futures to complete and collect results
                    for future in as_completed(futures):
                        suffix_name, num_of_ocu_s, basic_result, ocu_batch_enrichment, batch_indices = future.result()
                        self.logger.debug(f"Enrichment dictionary for {num_of_ocu_s}: {ocu_batch_enrichment}")

                        ocu_intermediate_path = os.path.join(self.intermediate_results_path, f"{batch[0]}-{batch[-1]}")

                        if num_of_ocu_s not in pcc_shuffle_arrays[suffix_name]:
                            pcc_shuffle_arrays[suffix_name][num_of_ocu_s] = {}
                            taxa_shuffled_arrays[suffix_name][num_of_ocu_s] = {}
                            shuffle_median_dict[suffix_name][num_of_ocu_s] = {}
                            p_values_bootstrap_dict[suffix_name][num_of_ocu_s] = {}
                            p_values_dict[suffix_name][num_of_ocu_s] = {}
                            gev_parameters_dict[suffix_name][num_of_ocu_s] = {}
                            for level in CONFIDENCE_LEVELS:
                                shuffle_ci_dict[level][suffix_name][num_of_ocu_s] = {}
                            ocu_dictionary_shuffle_enrichment[num_of_ocu_s] = {}

                        pcc_shuffle_arrays[suffix_name][num_of_ocu_s] = extend_instance(
                            pcc_shuffle_arrays[suffix_name][num_of_ocu_s], basic_result["pcc_shuffle_arrays"]
                        )
                        save_file(
                            pcc_shuffle_arrays, CORRELATION_COEFFICIENT_SHUFFLE_ARRAYS_FILE_NAME, ocu_intermediate_path
                        )

                        taxa_shuffled_arrays[suffix_name][num_of_ocu_s] = extend_instance(
                            taxa_shuffled_arrays[suffix_name][num_of_ocu_s],
                            basic_result["taxa_shuffle_array_tuples"]
                        )
                        save_file(taxa_shuffled_arrays, TAXA_SHUFFLED_ARRAYS_FILE_NAME, ocu_intermediate_path)

                        shuffle_median_dict[suffix_name][num_of_ocu_s] = extend_instance(
                            shuffle_median_dict[suffix_name][num_of_ocu_s], basic_result["pcc_shuffle_median"]
                        )
                        save_file(shuffle_median_dict, SHUFFLE_MEDIAN_FILE_NAME, ocu_intermediate_path)

                        p_values_bootstrap_dict[suffix_name][num_of_ocu_s] = extend_instance(
                            p_values_bootstrap_dict[suffix_name][num_of_ocu_s], basic_result["p_values_bootstrap"]
                        )
                        save_file(p_values_bootstrap_dict, BOOTSTRAP_P_VALUE_FILE_NAME, ocu_intermediate_path)
                        p_values_dict[suffix_name][num_of_ocu_s] = extend_instance(
                            p_values_dict[suffix_name][num_of_ocu_s], basic_result["p_values"]
                        )
                        save_file(p_values_dict, P_VALUES_FILE_NAME, ocu_intermediate_path)

                        gev_parameters_dict[suffix_name][num_of_ocu_s] = extend_instance(
                            gev_parameters_dict[suffix_name][num_of_ocu_s], basic_result["gev_parameters"]
                        )
                        save_file(gev_parameters_dict, GEV_PARAMETERS_FILE_NAME, ocu_intermediate_path)

                        for level in CONFIDENCE_LEVELS:
                            shuffle_ci_dict[level][suffix_name][num_of_ocu_s] = extend_instance(
                                shuffle_ci_dict[level][suffix_name][num_of_ocu_s],
                                basic_result[f"rho_shuffle_ci_{level}"]
                            )
                            save_file(shuffle_ci_dict[level], f"shuffle_ci_{level}.pkl", ocu_intermediate_path)

                        ocu_dictionary_shuffle_enrichment[num_of_ocu_s] = ocu_batch_enrichment

                        save_file(
                            ocu_dictionary_shuffle_enrichment,
                            SHUFFLES_ENRICHED_OCU_DICTIONARY_FILE_NAME,
                            ocu_intermediate_path
                        )
            except (BrokenProcessPool, OSError, ValueError):
                raise

            # Release memory after every batch
            pcc_shuffle_arrays.clear()
            taxa_shuffled_arrays.clear()
            shuffle_median_dict.clear()
            p_values_bootstrap_dict.clear()
            p_values_dict.clear()
            gev_parameters_dict.clear()
            shuffle_ci_dict.clear()
            ocu_dictionary_shuffle_enrichment.clear()

        self._combine_compores_basic_results(batches)

        self.combine_p_value_ranking(shuffling_cycle)

    def _combine_compores_basic_results(self, batch_structure: list[list]):

        # for every type of dictionary in batch_intermediate_path, combine them into one in ocu_intermediate_path
        self.logger.info("Combining batch dictionaries")
        self.combine_batch_dictionaries_to_ocu("pcc_shuffle_arrays", batch_structure)
        self.combine_batch_dictionaries_to_ocu("taxa_shuffled_arrays", batch_structure)
        self.combine_batch_dictionaries_to_ocu("shuffle_median", batch_structure)
        self.combine_batch_dictionaries_to_ocu("p_values_bootstrap", batch_structure)
        self.combine_batch_dictionaries_to_ocu("p_values", batch_structure)
        self.combine_batch_dictionaries_to_ocu("gev_parameters", batch_structure)
        for level in CONFIDENCE_LEVELS:
            self.combine_batch_dictionaries_to_ocu(f"shuffle_ci_{level}", batch_structure)

        self.combine_batch_dictionaries_to_ocu("ocu_dictionary_shuffles_enriched", batch_structure)

    def combine_p_value_ranking(self, shuffling_cycle_number: int) -> None:
        # calculate mean log p-value metric and save the result
        p_val_dict_temp = {}
        p_values_dict = self.load_dict(P_VALUES_FILE_NAME)
        for key in p_values_dict:
            p_val_dict_temp[key] = cast_nested_dict_to_array(p_values_dict[key])
        self.mean_log_p_value_dict = OneCaseCombination.mean_log_score_over_otu_clustering(p_val_dict_temp)
        save_file(self.mean_log_p_value_dict, MEAN_LOG_P_VALUE_FILE_NAME, self.intermediate_results_path)

        # sort responses by mean log p-value metric and save the ranking
        sorted_mean_log_p_value_df = pd.DataFrame(self.mean_log_p_value_dict, index=self.response_index)
        sorting_key = sorted_mean_log_p_value_df.columns.to_list()[0]
        sorted_mean_log_p_value_df = sorted_mean_log_p_value_df.sort_values(by=sorting_key, ascending=False)

        # add bootstrap p-values to the dataframe
        bootstrap_p_val_dict_temp = {}
        bootstrap_p_values_dict = self.load_dict(BOOTSTRAP_P_VALUE_FILE_NAME)
        for key in bootstrap_p_values_dict:
            bootstrap_p_val_dict_temp[key] = cast_nested_dict_to_array(bootstrap_p_values_dict[key])
        bootstrap_mean_log_p_value_dict = OneCaseCombination.mean_log_score_over_otu_clustering(
            bootstrap_p_val_dict_temp
        )
        save_file(
            bootstrap_mean_log_p_value_dict, BOOTSTRAP_MEAN_LOG_P_VALUE_FILE_NAME, self.intermediate_results_path
        )
        # concatenate bootstrap p-values to the p-value dataframe by response index
        bootstrap_mean_log_p_value_df = pd.DataFrame(bootstrap_mean_log_p_value_dict, index=self.response_index)
        sorted_mean_log_p_value_df = pd.concat([sorted_mean_log_p_value_df, bootstrap_mean_log_p_value_df], axis=1)
        # adjust column names
        sorted_mean_log_p_value_df.columns = ["gev", "bootstrap"]

        sorted_mean_log_p_value_path = str(os.path.join(
            self.outputs_path, RESPONSE_RANKING, sorting_key, self.coda_method
        ))
        os.makedirs(sorted_mean_log_p_value_path, exist_ok=True)
        with open(
                os.path.join(
                    sorted_mean_log_p_value_path,
                    f"minus_mean_log_p_value_sorted_responses_at_shuffling_cycle_{shuffling_cycle_number}.csv"
                ),
                'w'
        ) as f:
            sorted_mean_log_p_value_df.to_csv(f)

    def combine_batch_dictionaries_to_ocu(self, dict_name: str, batch_structure: list[list]):

        ocu_dict = {}
        for batch in batch_structure:
            batch_dict = self.load_dict(f"{dict_name}.pkl", f"{batch[0]}-{batch[-1]}")
            for key in batch_dict:
                if key not in ocu_dict:
                    ocu_dict[key] = {}
                for sub_key in batch_dict[key]:
                    if sub_key not in ocu_dict[key]:
                        ocu_dict[key][sub_key] = {}
                    for response_key in batch_dict[key][sub_key]:
                        ocu_dict[key][sub_key][response_key] = batch_dict[key][sub_key][response_key]

        if dict_name == "ocu_dictionary_shuffles_enriched":
            ocu_dict = dict(sorted(ocu_dict.items(), key=lambda x: x[0], reverse=True))
            for key in ocu_dict:
                for sub_key in ocu_dict[key]:
                    ocu_dict[key][sub_key] = dict(sorted(ocu_dict[key][sub_key].items(), key=lambda x: x[0]))
        else:
            for key in ocu_dict:
                ocu_dict[key] = dict(sorted(ocu_dict[key].items(), key=lambda x: x[0], reverse=True))
                for sub_key in ocu_dict[key]:
                    ocu_dict[key][sub_key] = dict(sorted(ocu_dict[key][sub_key].items(), key=lambda x: x[0]))

        save_file(ocu_dict, f"{dict_name}.pkl", self.intermediate_results_path)

        ocu_dict.clear()

    def run(self) -> None:
        """
        This function runs the analysis over one case combination
        """

        if self.step == 1:

            self.path_to_prepared_response = self.path_to_prepared_response.replace(".tsv", ".parquet")

            self.extract_response_tags()

            try:
                self.pool_run_compores_over_preprocessed_case()
            except (BrokenProcessPool, OSError, ValueError):
                raise

            if self.plotting_flag:
                self.logger.info("Combining composition response correlation signal visualization")
                plot_correlation_signal_significance_over_ocus(
                    self.intermediate_results_path,
                    self.significance_plots_path,
                    self.coda_method,
                    0
                )
            if self.n_shuffles == 0 or self.shuffle_cycles == 0:
                self.logger.info("Zero shuffles: finished w/o response correlation signal significance visualization.")
            self.update_state('run_comp', True)
            self.step += 1
            gc.collect()

        if self.step == 2:

            self.path_to_prepared_response = self.path_to_prepared_response.replace(".tsv", ".parquet")

            while len(self.state[f'{self.s1}-{self.s2}-{self.s3}']['run_comp_shuffle_iter']) < self.shuffle_cycles:
                num_cycles_done = len(self.state[f'{self.s1}-{self.s2}-{self.s3}']['run_comp_shuffle_iter'])
                if len(self.state[f'{self.s1}-{self.s2}-{self.s3}']['significance_viz']) == num_cycles_done:
                    if self.n_shuffles > 0:
                        try:
                            self.pool_run_compores_shuffles_over_preprocessed_case(num_cycles_done + 1)
                        except (BrokenProcessPool, OSError, ValueError):
                            raise
                        self._combine_and_store_ocu_level_results()
                        # self._combine_and_store_ocu_level_shuffle_results()
                    self.update_state('run_comp_shuffle_iter', self.n_shuffles)
                    num_cycles_done = len(self.state[f'{self.s1}-{self.s2}-{self.s3}']['run_comp_shuffle_iter'])

                    if self.n_shuffles > 0:
                        if self.plotting_flag:
                            self.logger.info(
                                f"Combining composition response correlation signal significance visualization"
                                f" after shuffling cycle {num_cycles_done}.")
                            plot_correlation_signal_significance_over_ocus(
                                self.intermediate_results_path,
                                self.significance_plots_path,
                                self.coda_method,
                                num_cycles_done,
                                self.n_shuffles
                            )
                self.update_state('significance_viz', True)

        if self.step == 3:
            if len(self.state[f'{self.s1}-{self.s2}-{self.s3}']['significance_viz']) < self.shuffle_cycles:
                if self.n_shuffles > 0:
                    if self.plotting_flag:
                        self.logger.info(f"Combining composition response correlation signal significance"
                                         f" visualization after shuffling cycle {self.shuffle_cycles}.")
                        plot_correlation_signal_significance_over_ocus(
                            self.intermediate_results_path,
                            self.significance_plots_path,
                            self.coda_method,
                            self.shuffle_cycles,
                            self.n_shuffles
                        )
                self.update_state('significance_viz', True)

        # Bring to the required format the keys in the final shuffle enrichment to the ocu dictionary
        ocu_dictionary_shuffle_enriched = self.load_dict(SHUFFLES_ENRICHED_OCU_DICTIONARY_FILE_NAME)
        ocu_dictionary_shuffle_enriched = {f"{k} OCUs": v for k, v in ocu_dictionary_shuffle_enriched.items()}
        save_file(
            ocu_dictionary_shuffle_enriched, SHUFFLES_ENRICHED_OCU_DICTIONARY_FILE_NAME, self.intermediate_results_path
        )

        # sort responses by mean log p-value metric and send the top results to the logger
        sorted_mean_log_p_value_df = pd.DataFrame(self.mean_log_p_value_dict, index=self.response_index)
        sorting_key = sorted_mean_log_p_value_df.columns.to_list()[0]
        sorted_mean_log_p_value_df = sorted_mean_log_p_value_df.sort_values(by=sorting_key, ascending=False)
        self.logger.info(
            f"Top signal mean log p-values:\n {sorted_mean_log_p_value_df.head(10)}. Check the output folder for more."
        )
        self.logger.info(f"Signal mean RMSE:\n {pd.DataFrame(self.mean_rmse_dict, index=self.response_index)}")


class ComporesMain:
    def __init__(self, config_file_path: Union[Path, str, AnyStr], ocu_case: int = None, deduplicate: bool = True):
        self.config_file_path = config_file_path
        self.config_dict = load_configuration(self.config_file_path)
        try:
            self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS)
            os.makedirs(self.log_files_path, exist_ok=True)
        except (ValueError, NotADirectoryError, PermissionError, KeyError, TypeError) as e:
            raise type(e)(f"Check the `PATH_TO_OUTPUTS` parameter value in the config file: {e}")
        self._initiate_logger(self.log_files_path)
        self.logger = self.logger_instance.get_logger()
        try:
            self.check_existing_input_paths()
        except FileNotFoundError as e:
            self.logger.error(f"Input file not found: {e}")
            sys.exit(1)
        self.balance_methods = ['CLR']
        if self.config_dict["CODA_METHOD"] and self.config_dict["CODA_METHOD"] != '':
            self.balance_methods = self.balance_methods + [self.config_dict["CODA_METHOD"]]
        self.correlation_type = self.config_dict["CORR"]
        self.shuffle_cycles = self.config_dict["SHUFFLE_CYCLES"]
        self.ocu_sampling_rate = self.config_dict["OCU_SAMPLING_RATE"]
        self.g1 = self.config_dict["GROUP1"]
        self.g2 = self.config_dict["GROUP2"]
        self.g3 = self.config_dict["GROUP3"]
        (
            self.path_to_microbiome, self.path_to_response, self.meta_data, self.ocu_clustering_results_path,
            self.path_to_sparcckit_res, self.path_to_sparcckit_corr,
            self.path_to_sparcckit_cov, self.path_to_microbiome_clustering, self.path_to_prepared_response
        ) = self.set_paths()
        self.sparcckit_iter = self.config_dict["SPARCCKIT_MAX_ITER"]
        self.imputed_samples_dictionary = {}
        self.clustered_ocu_dictionary = {}
        self.n_workers = self.set_n_workers()
        self.combinations_processed = 0
        self.results = {}
        self.plotting_flag = True
        self.ocu_case = ocu_case
        self.deduplicate = deduplicate
        self.ocu_limit = int(self.config_dict["MAX_OCU"])
        self.outlier_removal_flag = self.config_dict["OUTLIERS_REMOVAL"]

    @staticmethod
    def _get_total_cpu_count() -> int:
        """
        This function returns the number of CPUs available for the task.
        """
        if "SLURM_CPUS_PER_TASK" in os.environ:
            # SLURM environment variable for the number of CPUs allocated per task
            cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            # Fallback to the default CPU count on non-SLURM systems
            cpus_per_task = os.cpu_count()

        return cpus_per_task

    def set_n_workers(self) -> int:
        """
        This function sets the number of workers to a value defined in the configuration file or, in case it is not
        defined, to half of the number of CPUs available on the machine OR to 1 if the number of available CPUs is
        lower than 2. It also takes into account the SLURM environment variable for the number of CPUs allocated per
        task.
        """
        cpus_per_task = self._get_total_cpu_count()
        self.logger.info(f"There are {cpus_per_task} available CPUs for the task")
        n_workers = self.config_dict["N_WORKERS"]
        if n_workers is None or n_workers == '':
            n_workers = max(2, (cpus_per_task // 2 // 4) * 4) if cpus_per_task > 2 else 1
        self.logger.info(f"Number of workers for multiprocessing is set to {n_workers}")
        return n_workers

    def switch_off_plotting(self) -> None:
        """
        This function checks if the plotting flag should be set to False and updates it accordingly.
        :return: None
        """
        self.plotting_flag = False

    def _initiate_logger(self, path_to_logger: str, filemode: str = 'a') -> None:
        """
        Initialize the logger with the provided path.
        :return: None
        """
        self.logger_instance = CompoResLogger(log_name=__name__, filemode=filemode)
        self.logger_instance.update_logger_file_handler(os.path.join(path_to_logger, 'compores_sessions.log'))

    def get_balance_methods(self) -> list:
        """
        This function fetches the balance methods based on the configuration file, 'CLR' included by default.

        :return: The balance method list
        """
        return self.balance_methods

    def get_correlation_type(self) -> str:
        """
        This function fetches the correlation type from the configuration file.

        :return: The correlation problem type
        """
        return self.correlation_type

    def get_config_dict(self) -> dict:
        """
        This function fetches the configuration dictionary.

        :return: The configuration dictionary
        """
        return self.config_dict

    def get_config_file_path(self) -> str:
        """
        This function fetches the configuration file path.

        :return: The configuration file path
        """
        return self.config_file_path

    def get_results(self) -> dict:
        """
        This function fetches the result dictionary.

        :return: The result dictionary
        """
        return self.results

    def set_paths(self) -> tuple:
        """
        This function sets the paths to the files that are used by the Preprocessor instance.
        """
        s1 = self.g1
        s2 = self.g2
        s3 = self.g3
        path_to_preprocessing_results = str(os.path.join(self.config_dict["PATH_TO_OUTPUTS"], PREPROCESSING_RESULTS))
        path_to_sparcckit = str(os.path.join(path_to_preprocessing_results, 'sparcckit'))
        return (
            os.path.join(self.config_dict["PATH_TO_MICROBIOME"], f"{s1}-{s2}-{s3}.tsv"),
            os.path.join(self.config_dict["PATH_TO_RESPONSE"], f"{s1}-{s2}.tsv"),
            os.path.join(self.config_dict["PATH_TO_METADATA"], f"{s1}-{s2}-metadata.tsv"),
            os.path.join(self.config_dict["PATH_TO_OUTPUTS"], PREPROCESSING_RESULTS, 'microbiome', 'OCUs'),
            path_to_sparcckit,
            os.path.join(path_to_sparcckit, f"taxa_correlation_{s1}-{s2}-{s3}.tsv"),
            os.path.join(path_to_sparcckit, f"taxa_covariance_{s1}-{s2}-{s3}.tsv"),
            os.path.join(path_to_preprocessing_results, 'microbiome', f"{s1}-{s2}-{s3}.tsv"),
            os.path.join(path_to_preprocessing_results, 'response', f"{s1}-{s2}.tsv"),
        )

    def _prepare_list_of_ocu_folders_to_run_over(self, intermediate_results_path):
        """Prepares a list of OCU folders to run over as resulted at the preprocessing step."""
        s1 = self.g1
        s2 = self.g2
        s3 = self.g3
        suffix_name = f"{s1}-{s2}-{s3}"

        base_path = str(os.path.join(self.ocu_clustering_results_path, suffix_name))
        with open(os.path.join(base_path, f'{suffix_name}_ocu_clustering_dictionary.json'), 'r') as f:
            self.clustered_ocu_dictionary = json.load(f)
        if self.plotting_flag:
            sub_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            # sort by number value and not lexicographically
            sub_folders = sorted(sub_folders, key=lambda x: int(x))
            # from the OTU number down to MIN_OCU_NUM; sample every ocu_sampling_rate folder to dilute calculations
            sub_folders = sub_folders[::-1]
        else:
            sub_folders = [self.ocu_case]

        # check the number of taxa

        if int(sub_folders[0]) > self.ocu_limit:
            self.logger.warning(
                f"The number of taxa / OCUs in the microbiome file is {sub_folders[0]}. "
                f"The `MAX_OCU` parameter is configured to {self.ocu_limit} taxa / OCUs. "
                f"The analysis for the CLR CoDA method is currently running without limitation; "
                f"the `MAX_OCU` parameter is applied to the rest of the CoDA methods in the configuration file."
            )
        # store the sampled cluster ocu dictionary keys
        self.sampled_cluster_ocu_dictionary_keys = sub_folders
        save_file(self.sampled_cluster_ocu_dictionary_keys, SAMPLED_OCU_CASES_FILE_NAME, intermediate_results_path)

    def _prepare_filtered_copy_of_ocu_dictionary_to_enrich(self, intermediate_results_path):
        """ After the OCU folders are sampled, the function filters the initial OCU dictionary to only include
        information on the sampled OCUs to enrich it with the results on the calculations down the stream.
        """
        selected_keys = [f"{ocu_num} OCUs" for ocu_num in self.sampled_cluster_ocu_dictionary_keys]
        self.ocu_dictionary = {
            k: self.clustered_ocu_dictionary[k] for k in selected_keys if k in self.clustered_ocu_dictionary
        }
        for key in self.ocu_dictionary:
            for result_key in ["NUM_OCU", "DEN_OCU", "rho", "rmse"]:
                self.ocu_dictionary[key][result_key] = {}
        save_file(self.ocu_dictionary, OCU_DICTIONARY_FILE_NAME, intermediate_results_path)

    def update_state_to_preprocessed(self, value: Any, state: dict, coda_method: str):
        if f'{self.g1}-{self.g2}-{self.g3}' not in state:
            state[f'{self.g1}-{self.g2}-{self.g3}'] = {
                'preprocessed': False,
                'run_comp': False,
                'run_comp_shuffle_iter': [],
                'significance_viz': [],
                'otu_cumulative_p_value': False
            }

        state[f'{self.g1}-{self.g2}-{self.g3}']['preprocessed'] = value
        with open(os.path.join(self.config_dict["PATH_TO_OUTPUTS"], f'state_{coda_method}.json'), 'w') as f:
            json.dump(state, f, indent=4)

    def check_existing_input_paths(self) -> None:
        """
        This function checks if the paths to the input data in the configuration file exist.

        :return: None
        """
        # Check if the PATH_TO_MICROBIOME in the configuration file exist
        if not os.path.exists(self.config_dict["PATH_TO_MICROBIOME"]):
            raise FileNotFoundError(f'The path {self.config_dict["PATH_TO_MICROBIOME"]} does not exist.')

        # Check if the PATH_TO_RESPONSE in the configuration file exist
        if not os.path.exists(self.config_dict["PATH_TO_RESPONSE"]):
            raise FileNotFoundError(f'The path {self.config_dict["PATH_TO_RESPONSE"]} does not exist.')

    def close(self) -> None:
        """Closes instance attributes"""
        if self.logger_instance and hasattr(self, 'logger_instance'):
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        # Clear if other attributes are added that can keep handlers running or memory used

    def load_state(self, suffix_1: str, suffix_2: str, suffix_3: str, balance_method: str) -> dict:
        try:
            with open(os.path.join(self.config_dict["PATH_TO_OUTPUTS"], f'state_{balance_method}.json'), 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                f'{suffix_1}-{suffix_2}-{suffix_3}': {
                    'preprocessed': False,
                    'run_comp': False,
                    'run_comp_shuffle_iter': [],
                    'significance_viz': [],
                    'otu_cumulative_p_value': False
                }
            }

    @staticmethod
    def fetch_first_response_tag(path_to_ranking, n_shuffle_cycles):

        response_tag = ''
        for file in os.listdir(path_to_ranking):
            if file.endswith(f"{n_shuffle_cycles}.csv"):
                response_rankings = pd.read_csv(os.path.join(path_to_ranking, file))
                response_tag = response_rankings.iloc[0, 0]

        return response_tag

    def run(self, only_preprocess_step_flag: bool = False) -> None:  # TODO: check the former name `main`
        """
        This is the main function of the program running calculations over all requested cases.

        :return: None
        """

        self.logger.info("Started CompoRes data processing.")
        suffix_name = f'{self.g1}-{self.g2}-{self.g3}'

        # Update the logger file handler to a specific case location
        self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS, suffix_name)
        os.makedirs(self.log_files_path, exist_ok=True)
        self.logger_instance.update_logger_file_handler(
            os.path.join(self.log_files_path, 'compores_main.log')
        )

        # Update the logger file handler to a specific case location
        self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS, suffix_name)
        os.makedirs(self.log_files_path, exist_ok=True)
        self.logger_instance.update_logger_file_handler(
            os.path.join(self.log_files_path, 'compores_main.log')
        )
        for balance_method in self.balance_methods:
            # check the state of the case
            current_state = self.load_state(self.g1, self.g2, self.g3, balance_method)
            self.logger.info(f"Current state: {current_state}")
            if suffix_name in current_state:
                if len(current_state[suffix_name]['run_comp_shuffle_iter']) < self.shuffle_cycles:
                    if not current_state[suffix_name]['run_comp']:
                        if not current_state[suffix_name]['preprocessed']:
                            run_step = 0
                        else:
                            run_step = 1
                    else:
                        run_step = 2  # the case of a lacking plotting step is treated in the runner
                else:
                    run_step = 3
            else:
                run_step = 0

            if run_step == 0:
                intermediate_results_path = os.path.join(
                    self.config_dict["PATH_TO_OUTPUTS"], BASIC_RESULTS,
                    f'{self.g1}-{self.g2}-{self.g3}', balance_method)
                response_vs_balance_plots_path = os.path.join(
                    self.config_dict["PATH_TO_OUTPUTS"], PLOT_PATH, RESPONSE_VS_BEST_BALANCE_PATH,
                    f'{self.g1}-{self.g2}-{self.g3}'
                )
                if balance_method == 'CLR':
                    preprocessor = Preprocessor(
                        self.logger,
                        self.g1,
                        self.g2,
                        self.g3,
                        self.path_to_microbiome,
                        self.path_to_response,
                        self.path_to_microbiome_clustering,
                        self.path_to_prepared_response,
                        self.path_to_sparcckit_res,
                        self.path_to_sparcckit_corr,
                        self.path_to_sparcckit_cov,
                        self.config_dict["PATH_TO_OUTPUTS"],
                        self.ocu_clustering_results_path,
                        response_vs_balance_plots_path,
                        self.balance_methods,
                        self.sparcckit_iter,
                        self.ocu_sampling_rate,
                        deduplicate_flag=self.deduplicate,
                        regular_run_flag=self.plotting_flag,
                        outlier_removal_tag=self.outlier_removal_flag,
                        path_to_metadata=self.meta_data,
                        config_dict=self.config_dict,
                    )
                    try:
                        preprocessor.process()
                    except (
                            MisMatchFiles, NonNumericDataFrameError, NegativeValuesDataFrameError,
                            EmptyDataFrame, MinDataFrame, DuplicatedIndices, ValueError, OutlierCheckFailed
                    ) as e:
                        self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS)
                        self.logger_instance.update_logger_file_handler(
                            os.path.join(self.log_files_path, "compores_sessions.log")
                        )
                        self.logger.error(f"{e}")
                        raise
                    self.imputed_samples_dictionary = preprocessor.get_imputed_samples_dictionary()
                    self.clustered_ocu_dictionary = preprocessor.get_ocu_clustering_dictionary()
                self._prepare_list_of_ocu_folders_to_run_over(intermediate_results_path)
                self._prepare_filtered_copy_of_ocu_dictionary_to_enrich(intermediate_results_path)
                self.update_state_to_preprocessed(True, current_state, balance_method)
                run_step += 1

            if only_preprocess_step_flag:
                if len(self.balance_methods) > 1:
                    for b_m in self.balance_methods[1:]:
                        intermediate_results_path = os.path.join(
                            self.config_dict["PATH_TO_OUTPUTS"],
                            BASIC_RESULTS, f"{self.g1}-{self.g2}-{self.g3}", b_m,
                        )
                        self._prepare_list_of_ocu_folders_to_run_over(intermediate_results_path)
                        self._prepare_filtered_copy_of_ocu_dictionary_to_enrich(intermediate_results_path)
                        self.update_state_to_preprocessed(True, current_state, b_m)
                self.logger.info("Preprocessing step only: exiting before running CoDa analysis.")
                return

            combination = OneCaseCombination(
                self.logger, self.config_dict, self.plotting_flag, self.g1, self.g2, self.g3, run_step, self.n_workers,
                balance_method, self.ocu_case, self.deduplicate
            )
            try:
                combination.run()
                self.combinations_processed += 1
                # Return the logger to the original file handler
                self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS)
                self.logger_instance.update_logger_file_handler(
                    os.path.join(self.log_files_path, 'compores_sessions.log'))
                self.logger.info(f"CompoRes data processing completed; {self.combinations_processed} cases processed.")
            except (BrokenProcessPool, OSError, FileNotFoundError, ValueError) as e:
                # Return the logger to the session file handler
                self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS)
                self.logger_instance.update_logger_file_handler(
                    os.path.join(self.log_files_path, 'compores_sessions.log'))
                self.logger.error(
                    f"{e}: If not intentionally canceled, the job may require one of the following:"
                    f" checking the consistency of the input data, raising memory/time limit, reducing the input data"
                    f" dimensionality, or checking disk space availability; in case you need to process multiple"
                    f" experiments, make sure every case is run in a separate job;"
                    f" some intermediate results might have been lost;"
                    f" re-run the job after investigating the issue and implementing relevant adjustments.")
                raise
            except (MisMatchFiles, NonNumericDataFrameError, NegativeValuesDataFrameError,
                    EmptyDataFrame, MinDataFrame, DuplicatedIndices) as e:
                self.log_files_path = os.path.join(self.config_dict["PATH_TO_OUTPUTS"], LOGS)
                self.logger_instance.update_logger_file_handler(
                    os.path.join(self.log_files_path, 'compores_sessions.log'))
                self.logger.error(f"{e}")
                raise

    def generate_otu_p_value_summary_data(self, partial_response_label: str | None = None) -> None:
        if self.plotting_flag:
            self.logger.info("Starting OTU-wise cumulative p-value analysis.")
            microbiome_case = f'{self.g1}-{self.g2}-{self.g3}'

            try:
                otu_tracing_state, response_label = self._sync_on_otu_tracing_state(partial_response_label)
            except NoResponseLabelFound as e:
                self.logger.error(f"{e} Exiting.")
                sys.exit(1)

            if not otu_tracing_state:
                for balance_method in self.balance_methods:
                    self.logger.info(f"Performing OTU-wise cumulative p-value analysis for {balance_method}.")
                    filtered_responses = self._get_response_info(balance_method, microbiome_case, response_label)
                    tasks = [
                        (
                            self.config_dict, self.g1, self.g2, self.g3, balance_method, i, r
                        ) for i, r in filtered_responses
                    ]
                    with ProcessPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(self._process_single_response, *task): task for task in tasks}
                        for future in as_completed(futures):
                            task = futures[future]
                            try:
                                result = future.result()
                                self.logger.info(result)
                            except Exception as e:
                                self.logger.error(
                                    f"OTU tracing failed: unknown CoDA method or missing file; {task[6]}: {e}"
                                )
                                sys.exit(1)
                    self._update_otu_tracing_state(balance_method)
                    self.logger.info(f"Finished OTU cumulative p-value analysis for {balance_method}.")
            self.logger.info("OTU cumulative p-value analysis completed.")
        else:
            self.logger.info(
                "OTU-wise cumulative p-value analysis has been already performed."
            )

    @staticmethod
    def _process_single_response(
            config_dict: dict, g1: str, g2: str, g3: str, balance_method: str, res_index: int, res_tag: str
    ):
        p_value_object = ComporesClusteredPValueCalculations(config_dict, g1, g2, g3)
        p_value_object.set_current_response(res_index, res_tag, balance_method)
        try:
            p_value_object.build_otu_p_value_matrix(balance_method)
            p_value_object.prepare_final_otu_tracing_output()
            if balance_method == 'CLR':
                return f"Generated a matrix of OTU-wise cumulative p-value metrics for CLR: {res_tag}."
            elif balance_method == 'pairs':
                return f"Generated a matrix of OTU-pairwise cumulative p-value metrics for pairs balance: {res_tag}."

        except ValueError or FileNotFoundError as e:
            raise e

    def _update_otu_tracing_state(self, balance_method):
        p_value_object = ComporesClusteredPValueCalculations(self.config_dict, self.g1, self.g2, self.g3)
        if balance_method == 'CLR':
            p_value_object.update_otu_cumulative_p_value_analysis_state('CLR', True)
        elif balance_method == 'pairs':
            p_value_object.update_otu_cumulative_p_value_analysis_state('pairs', True)

    def _sync_on_otu_tracing_state(self, response_tag_start: str | None = None):
        p_value_object = ComporesClusteredPValueCalculations(
            self.config_dict, self.g1, self.g2, self.g3, True, response_tag_start
        )
        return p_value_object.state, p_value_object.response_label

    def _get_response_info(self, balance_method, microbiome_case, response_label):
        path_to_intermediate_results = os.path.join(
            self.config_dict["PATH_TO_OUTPUTS"], BASIC_RESULTS, microbiome_case,
            balance_method
        )
        try:
            if response_label is not None:
                response_label, response_labels = fetch_full_target_response_label(
                    path_to_intermediate_results, response_label
                )
                filtered_responses = [(i, r) for (i, r) in enumerate(response_labels) if r == response_label]
            else:
                response_labels = load_file(RESPONSE_INDEX_FILE_NAME, path_to_intermediate_results)
                filtered_responses = [(i, r) for (i, r) in enumerate(response_labels)]
            return filtered_responses

        except NoResponseLabelFound as e:
            self.logger.error(f"{e} Exiting.")
            sys.exit(1)

    def add_synthetic_data_analysis(self, response_label: str | None) -> None:

        if response_label:
            self.logger.info(f"Starting derived synthetic classification power analysis for: '{response_label}'.")
        else:
            self.logger.info("Starting derived synthetic classification power analysis.")

        binary_classification_path = f'{SYNTHETIC_DATA_PATH}/analyze_binary_classification'
        # sys.path.append(SYNTHETIC_DATA_PATH)
        # sys.path.append(binary_classification_path)

        current_file = os.path.abspath(__file__)
        current_dir = os.path.join(os.path.dirname(os.path.dirname(current_file)), binary_classification_path)
        auroc_curve_script_path = os.path.join(current_dir, 'analyze_results_using_synthetic_data.sh')
        # Check if the script path exists
        if not os.path.exists(auroc_curve_script_path):
            self.logger.info(f"Script path does not exist: {auroc_curve_script_path}. Exiting.")
            sys.exit(1)

        # Define paths
        path_to_outputs = self.config_dict["PATH_TO_OUTPUTS"]
        directory_to_preprocessed_microbiome = str(os.path.join(path_to_outputs, PREPROCESSING_RESULTS, "microbiome"))

        # TODO rename coda_method (the term itself means the same as balance_method)
        if 'pairs' in self.balance_methods:
            balance_method = 'pairs'
            coda_method = 'balance'
        else:
            balance_method = 'CLR'
            coda_method = 'taxon'

        microbiome_case = f'{self.g1}-{self.g2}-{self.g3}'
        microbiome_path = os.path.join(directory_to_preprocessed_microbiome, f'{microbiome_case}.tsv')
        microbiome = pd.read_csv(str(microbiome_path), sep='\t', header=0, index_col=0)
        num_otus = microbiome.shape[1]
        sample_size = microbiome.shape[0]
        response_labels_path = os.path.join(path_to_outputs, BASIC_RESULTS, microbiome_case, balance_method)
        response_ranking_path = os.path.join(path_to_outputs, RESPONSE_RANKING, microbiome_case, balance_method)

        if response_label:
            try:
                response_label, _ = fetch_full_target_response_label(response_labels_path, response_label)
            except NoResponseLabelFound as e:
                self.logger.error(f"{e} Exiting.")
                sys.exit(1)
        else:
            response_label = self.fetch_first_response_tag(response_ranking_path, self.shuffle_cycles)

        response_data_input, response_label = fetch_synthetic_analysis_input_data(
            path_to_outputs, microbiome_case, balance_method, self.ocu_limit, response_label
        )

        # Check if response_data_input has more than 3 items
        if len(response_data_input["rmse"]) < 3:
            self.logger.info(
                f"Not enough response data points ({len(response_data_input['rmse'])}) for "
                f"synthetic power analysis for '{response_label}'. Need at least 3. Try re-running `CompoRes` "
                f"after balancing `TAXA_FILTER` and `OUTLIERS_REMOVAL` parameters. Exiting."
            )
            sys.exit(1)

        initial_rmse_values = np.round(response_data_input["rmse"], 3)
        response_data_input = deduplicate_synthetic_analysis_input_data(response_data_input)
        rmse_values = response_data_input["rmse"]
        slopes = response_data_input["slope"]
        intercepts = response_data_input["intercept"]
        num_ocu_s = response_data_input["num_ocu"]
        den_ocu_s = response_data_input["den_ocu"]
        rmse_ocu_number = response_data_input["ocu_number"]
        dir_to_save_results = os.path.join(path_to_outputs, SYNTHETIC_ANALYSIS_RESULTS_PATH, response_label)
        initial_response_values_str = f'"{" ".join(map(str, initial_rmse_values))}"'
        response_noise_values_str = f'"{" ".join(map(str, rmse_values))}"'
        slope_values_str = f'"{" ".join(map(str, slopes))}"'
        intercept_values_str = f'"{" ".join(map(str, intercepts))}"'
        rmse_ocu_number_str = f'"{" ".join(map(str, rmse_ocu_number))}"'
        num_ocu_s_str = '"'+';'.join(f'[{",".join(sublist)}]' for sublist in num_ocu_s)+'"'
        den_ocu_s_str = '"'+';'.join(f'[{",".join(sublist)}]' for sublist in den_ocu_s)+'"'

        # Combine the command into a single string and run the script
        command = ' '.join([
            'bash', auroc_curve_script_path, str(num_otus), str(sample_size),
            str(NUMBER_OF_RESPONSES_TO_GENERATE), balance_method,  # TODO: fix balance/coda/lr
            str(NUMBER_OF_EXPERIMENT_REPEATS), coda_method, str(self.ocu_sampling_rate), slope_values_str,
            intercept_values_str, num_ocu_s_str, den_ocu_s_str,
            microbiome_case, microbiome_path, dir_to_save_results, response_noise_values_str, rmse_ocu_number_str,
            initial_response_values_str, response_label, str(self.config_dict["TAXA_FILTER"]), str(self.sparcckit_iter),
            str(self.ocu_limit), self.correlation_type, str(self.config_dict["N_SHUFFLES"]), str(self.shuffle_cycles),
            current_dir
        ])
        # Run the command
        self.logger.info(f"Running command: {command}")
        if self.plotting_flag:
            capture_output_flag = False
        else:
            capture_output_flag = True
        subprocess.run(command, capture_output=capture_output_flag, text=True, shell=True)

        self.logger.info(
                f"Finished derived synthetic data and classification power analysis run for '{response_label}'."
            )
