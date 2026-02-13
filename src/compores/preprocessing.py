import json
import os
import sys
from logging import Logger
import csv
from pathlib import Path
from typing import Union, AnyStr, Any

import numpy as np
import pandas as pd
from numpy import floating
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm
from statsmodels import robust

from .exceptions_module import MisMatchFiles, NonNumericDataFrameError, NegativeValuesDataFrameError, \
    EmptyDataFrame, MinDataFrame, DuplicatedIndices, OutlierCheckFailed
from .utils import invert_dict, extract_response_tags
from .sparcckit import SparccKit

# CompoRes preprocessing path constants
PREPROCESSED_SAMPLES = 'preprocessed_samples'
PREPROCESSING_RESULTS = 'preprocessing_results'
OUTLIER_DETECTION = "outlier_detection"
SPARCCKIT_PATH = 'sparcckit'

MIN_OCU_NUM = 3
SPARCCKIT_MIN_ITER = 50
SPARCCKIT_MAX_ITER_SCALING_FACTOR = 5000

MIN_SAMPLES_FOR_OUTLIER_REMOVAL = 4
OUTLIER_ALPHA = 0.025


class Preprocessor:

    def __init__(
            self,
            logger: Logger,
            s1: str, s2: str, s3: str,
            path_to_microbiome: Union[Path, str, AnyStr],
            path_to_response: Union[Path, str, AnyStr],
            path_to_microbiome_clustering: Union[Path, str, AnyStr],
            path_to_prepared_response: Union[Path, str, AnyStr],
            path_to_sparcckit_results: Union[Path, str, AnyStr],  # TODO: do we need this path?
            path_to_sparcckit_corr: Union[Path, str, AnyStr],
            path_to_sparcckit_cov: Union[Path, str, AnyStr],
            path_to_outputs: Union[Path, str, AnyStr],
            path_to_clustered_ocu: Union[Path, str, AnyStr],
            path_to_response_vs_balance_plots: Union[Path, str, AnyStr],
            balance_methods_list: list[str],
            sparcckit_iter: int,
            clustering_sampling_rate: int,
            imputation_flag: bool = True,
            deduplicate_flag: bool = False,
            regular_run_flag: bool = False,
            outlier_removal_tag: bool = False,
            path_to_metadata: Union[Path, str, AnyStr] = "",
            config_dict: dict = None,
    ):
        self.logger = logger
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        # check if the file exists
        for file_path in [path_to_microbiome, path_to_response]:
            if not os.path.exists(file_path):
                error_msg = f"File `{file_path}` not found."
                self.logger.error(f"Error creating Preprocessor object: {error_msg}")
                sys.exit(1)
        self.path_to_microbiome = path_to_microbiome
        self.path_to_response = path_to_response
        self.path_to_microbiome_clustering = path_to_microbiome_clustering
        self.path_to_prepared_response = path_to_prepared_response
        self.outputs_path = path_to_outputs
        self.imputed_x = pd.DataFrame()
        self.imputed_samples_dict = {}
        self.clustered_ocu_path = path_to_clustered_ocu
        self.linkage_matrix = np.ndarray([])
        self.path_to_sparcckit_results = path_to_sparcckit_results  # TODO: do we need this path?
        self.path_to_sparcckit_input = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, SPARCCKIT_PATH)
        self.path_to_sparcckit_corr = path_to_sparcckit_corr
        self.path_to_sparcckit_cov = path_to_sparcckit_cov
        self.clustered_ocu_dictionary = {}
        self.file_name = f"{self.s1}-{self.s2}-{self.s3}"
        self.imputation_flag = imputation_flag
        self.path_response_vs_balance_plots = path_to_response_vs_balance_plots
        self.balance_methods = balance_methods_list
        self.sparcckit_iter = sparcckit_iter
        self.ocu_sampling_rate = clustering_sampling_rate
        self.deduplicate_flag = deduplicate_flag
        self.regular_run_flag = regular_run_flag
        self.outlier_removal_flag = outlier_removal_tag
        self.meta_data = path_to_metadata
        if config_dict is not None:
            self.config_dict = config_dict

    def get_imputed_samples_dictionary(self) -> dict:
        """
        Returns the dictionary of imputed samples with the corresponding taxa.
        :return: Dictionary of imputed samples with the corresponding taxa.
        """
        return self.imputed_samples_dict

    def get_imputed_data(self) -> pd.DataFrame:
        """
        Returns the imputed data matrix.
        :return: Imputed data matrix.
        """
        return self.imputed_x

    def get_ocu_clustering_dictionary(self) -> dict:
        """
        Returns the dictionary of imputed samples with the corresponding taxa.
        :return: Dictionary of imputed samples with the corresponding taxa.
        """
        return self.clustered_ocu_dictionary

    def process(self):
        self.logger.info("Preprocessing started.")

        try:
            self.check_input_files_for_duplicated_indices(self.path_to_microbiome, self.path_to_response)
        except DuplicatedIndices as e:
            self.logger.error(f"Error checking input files for duplicate indices: {e}")
            raise

        try:
            self.check_input_files_for_same_unique_indices(self.path_to_microbiome, self.path_to_response, step='raw')
        except (MisMatchFiles, MinDataFrame) as e:
            self.logger.error(f"Error checking input files for same indices: {e}")
            raise

        try:
            self.prepare_input("microbiome", self.path_to_microbiome, threshold=self.config_dict["TAXA_FILTER"])
            self.prepare_input("response", self.path_to_response)
        except (NonNumericDataFrameError, NegativeValuesDataFrameError, EmptyDataFrame, MinDataFrame, ValueError) as e:
            self.logger.error(f"Error preparing input: {e}")
            raise

        try:
            self.check_input_files_for_same_unique_indices(
                self.path_to_microbiome_clustering, self.path_to_prepared_response, step='preprocessed'
            )
        except (MisMatchFiles, MinDataFrame) as e:
            self.logger.error(f"Error checking input files for same indices: {e}")
            raise

        if self.outlier_removal_flag and self.regular_run_flag:
            try:
                self.perform_outlier_check_and_removal()
            except OutlierCheckFailed:
                raise

        os.makedirs(self.path_to_sparcckit_results, exist_ok=True)
        if self.regular_run_flag:
            microbiome_df = pd.read_csv(self.path_to_microbiome_clustering, sep="\t", index_col=0)
            sparcckit_obj = SparccKit(
                self.logger,
                microbiome_df,
                max_iter=self.calculate_max_iterations(microbiome_df.shape[1])
            )
            sparcckit_obj.run()
            sparcckit_corr = sparcckit_obj.get_corr_file()
            with open(self.path_to_sparcckit_corr, 'w') as f:
                sparcckit_corr.to_csv(f, sep="\t")
            message = "sparcckit correlation estimates file generated."
            self.logger.info(message)
        else:

            # find the microbiome tsv file 3 levels up from self.path_to_sparcckit_results
            file_name_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.path_to_sparcckit_results)))
            tsv_file = [f for f in os.listdir(file_name_folder) if f.endswith('.tsv')]
            # extract the base name of the microbiome tsv file
            if not tsv_file:
                error_msg = f"No microbiome file found in `{file_name_folder}`."
                self.logger.error(error_msg)
                sys.exit(1)
            else:
                tsv_file = tsv_file[0]

            # Copy the existing sparcckit correlation file to self.path_to_sparcckit_results
            source_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(self.path_to_sparcckit_results))))))  # 6 levels up from self.path_to_sparcckit_results
            source_folder = os.path.join(source_folder, PREPROCESSING_RESULTS, SPARCCKIT_PATH)
            source_file = os.path.join(source_folder, f"taxa_correlation_{tsv_file}")
            if os.path.exists(source_file):
                with open(source_file, 'r') as src, open(self.path_to_sparcckit_corr, 'w') as dst:
                    dst.write(src.read())
                message = f"Using existing sparcckit correlation estimates file from {source_file}."
                self.logger.info(message)
            else:
                error_msg = f"Source sparcckit correlation file `{source_file}` not found."
                self.logger.error(error_msg)
                sys.exit(1)

        self.create_linkage()
        self.save_ocu_matrices()

        self.logger.info("Preprocessing finished.")

    def calculate_max_iterations(self, microbiome_size: int) -> int:
        """
        Calculates the maximum number of iterations for SparccKit based on the number of taxa in the microbiome
        dataframe by scaling the base iteration count (self.sparcckit_iter) according to dataset width. The iteration
        count decays exponentially with the number of total taxa in microbiome_df, but is bounded to never fall below
        SPARCCKIT_MIN_ITER or exceed self.sparcckit_iter. The approach allows sufficient iterations while balancing
        computational complexity for extremely large datasets.

        :param microbiome_size: The width of the microbiome (the number of taxa).
        :return: The calculated maximum number of iterations for SparccKit.
        """
        return min(self.sparcckit_iter, max(
            SPARCCKIT_MIN_ITER, int(
                self.sparcckit_iter * np.exp(- microbiome_size / SPARCCKIT_MAX_ITER_SCALING_FACTOR))
        ))

    def perform_outlier_check_and_removal(self):
        """Performs an outlier check and removes samples based on their effect on the regression slopes of response
        variables on microbiome features. This method identifies common outlier samples and evaluates whether their
        removal causes significant changes in the downstream correlation analysis. If significant changes are detected,
        the outliers are removed from both microbiome and response datasets, and the updated datasets are saved. If no
        significant changes are observed, no outlier removal takes place. When categorical metadata is provided, outlier
        detection is performed within each category. Bonferroni correction is applied when multiple response variables
        are present for both outlier candidate samples detection and the significance testing of slope changes.

        :raises OutlierCheckFailed: When the number of samples remaining after outlier removal is lower than the
            minimum threshold `MIN_SAMPLES_FOR_OUTLIER_REMOVAL`.

        :return: None.
        """
        from .compores_main import OneCaseCombination

        common_response_mask = self.find_common_outlier_mask()
        if common_response_mask.sum() < MIN_SAMPLES_FOR_OUTLIER_REMOVAL:
            message = (f"Exited after outlier check: after dropping potential outliers, only "
                       f"{common_response_mask.sum()} samples remain, which is less than the required minimum of "
                       f"{MIN_SAMPLES_FOR_OUTLIER_REMOVAL}. Please check the masked response file in "
                       f"`{PREPROCESSING_RESULTS}/{OUTLIER_DETECTION}` folder; re-run upon reviewing the dataset.")
            raise OutlierCheckFailed(message)

        else:
            microbiome_df = pd.read_csv(self.path_to_microbiome_clustering, sep="\t", index_col=0)
            response_df = pd.read_csv(self.path_to_prepared_response, sep="\t", index_col=0)
            combination = OneCaseCombination(
                self.logger,
                self.config_dict,
                self.regular_run_flag,
                self.s1,
                self.s2,
                self.s3,
                ocu_case=self.file_name,
                deduplicate=self.deduplicate_flag,
            )
            combination.response_index = extract_response_tags(
                self.path_to_prepared_response, combination.intermediate_results_path
            )
            all_sample_slope_array, all_sample_se_array, n_taxa_dict = self.calculate_slopes_and_ses(
                combination, microbiome_df, response_df, True, True, False
            )
            masked_sample_slope_array, masked_sample_se_array, _ = self.calculate_slopes_and_ses(
                combination, microbiome_df[common_response_mask], response_df[common_response_mask], True, False, True,
                n_taxa_dict)

            change_array = self.calculate_slope_change_p_values(
                all_sample_slope_array,
                all_sample_se_array,
                masked_sample_slope_array,
                masked_sample_se_array,
                combination.response_index,
            )
            change_test = change_array < 2 * OUTLIER_ALPHA / response_df.shape[1]
            if change_test.any():
                list_of_samples_to_remove = common_response_mask[~common_response_mask].index.tolist()
                affected_responses = response_df.columns[change_test].tolist()
                self.logger.info(
                    f"Please check the masked response file in `{PREPROCESSING_RESULTS}/{OUTLIER_DETECTION}` "
                    f"folder. Samples {list_of_samples_to_remove} have been detected as outliers that "
                    f"affect {change_test.sum()} response variables out of {len(change_array)} "
                    f"in a statistically significant manner: {affected_responses}; they are now "
                    f"removed from both microbiome and response dataframes."
                )

                report_file_path = os.path.join(
                    self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION
                )
                report_file_name = os.path.join(report_file_path, "outlier_final_report.tsv")
                with open(report_file_name, "a") as report_file:
                    report_file.write(f"Outlier samples:\t{list_of_samples_to_remove}\n")
                    report_file.write(f"Number of outlier samples:\t{len(list_of_samples_to_remove)}\n")
                    report_file.write(f"Total samples:\t{common_response_mask.shape[0]}\n")
                    report_file.write(f"Affected responses:\t{affected_responses}\n")
                    report_file.write(f"Number of affected responses:\t{change_test.sum()}\n")
                    report_file.write(f"Total responses:\t{len(change_array)}\n")

                # Store updated microbiome and response datasets without outliers
                microbiome_df = microbiome_df[common_response_mask]
                response_df = response_df[common_response_mask]
                with open(self.path_to_microbiome_clustering, 'w') as f:
                    microbiome_df.to_csv(f, sep="\t")
                with open(self.path_to_prepared_response, "w") as f:
                    response_df.to_csv(f, sep="\t")
                with open(self.path_to_prepared_response.replace(".tsv", ".parquet"), 'wb') as f:
                    response_df.to_parquet(f, engine="fastparquet")

                # Store initial microbiome dataset without outliers
                raw_microbiome_df = pd.read_csv(self.path_to_microbiome, sep="\t", index_col=0)
                raw_microbiome_df = raw_microbiome_df[common_response_mask]
                masked_raw_microbiome_file_name = os.path.join(report_file_path, "raw_microbiome_wo_outliers.tsv")
                with open(masked_raw_microbiome_file_name, 'w') as f:
                    raw_microbiome_df.to_csv(f, sep="\t")
            else:
                self.logger.info("Outlier removal will NOT be applied: samples detected as potential outliers do not "
                                 "significantly affect the regression slopes of the response variables. You can "
                                 f"check the masked response file in `{PREPROCESSING_RESULTS}/{OUTLIER_DETECTION}`"
                                 f" folder.")

    def check_input_files_for_duplicated_indices(
            self, m_file_path: Union[Path, str], r_file_path: Union[Path, str]
    ) -> None:
        """Checks if the two input files have duplicated sample tag names.

        :param m_file_path: Microbiome file path.
        :param r_file_path: Response file path.
        :return: None
        """
        m_df = pd.read_csv(m_file_path, sep="\t", index_col=0)
        r_df = pd.read_csv(r_file_path, sep="\t", index_col=0)

        microbiome_file_name = os.path.basename(m_file_path)
        response_file_name = os.path.basename(r_file_path)

        if not self.deduplicate_flag:

            if m_df.index.duplicated().any():
                raise DuplicatedIndices(file_name=microbiome_file_name)

            if r_df.index.duplicated().any():
                raise DuplicatedIndices(file_name=response_file_name)

    def check_input_files_for_same_unique_indices(
            self, m_file_path: Union[Path, str], r_file_path: Union[Path, str], step: str
    ) -> None:
        """This function filters common samples for the two input files, checks the dimensionality of the resulting
        dataframes, and sorts them alphabetically.

        :param m_file_path: Microbiome file path.
        :param r_file_path: Response file path.
        :param step: Files checked, 'raw' or 'preprocessed'.
        :return: None
        """

        m_df_raw = pd.read_csv(m_file_path, sep="\t", index_col=0)
        r_df_raw = pd.read_csv(r_file_path, sep="\t", index_col=0)

        microbiome_file_name = os.path.basename(m_file_path)
        response_file_name = os.path.basename(r_file_path)

        if step == 'raw' and self.deduplicate_flag:
            if m_df_raw.index.duplicated().any():
                self.logger.info(f"Duplicated sample tag names in `{microbiome_file_name}`: using the first of the"
                                 f" duplicated sample tags: {m_df_raw[m_df_raw.index.duplicated()].index.to_list()}.")
                m_df_raw = m_df_raw[~m_df_raw.index.duplicated()]

            if r_df_raw.index.duplicated().any():
                self.logger.info(f"Duplicated sample tag names in `{response_file_name}`: using the first of the"
                                 f" duplicated sample tags: {r_df_raw[r_df_raw.index.duplicated()].index.to_list()}.")
                r_df_raw = r_df_raw[~r_df_raw.index.duplicated()]

        non_common_indexes = m_df_raw.index.symmetric_difference(r_df_raw.index)
        common_indexes = m_df_raw.index.intersection(r_df_raw.index)
        m_df_common = m_df_raw.loc[common_indexes].copy()
        r_df_common = r_df_raw.loc[common_indexes].copy()

        # check for a minimum of 3 samples and 3 taxa
        if any(size < 3 for size in m_df_common.shape):
            raise MinDataFrame(file_name=microbiome_file_name)
        if r_df_common.shape[0] < 3 or r_df_common.shape[1] < 1:
            raise MinDataFrame(file_name=response_file_name)

        m_df_common.sort_index(inplace=True)
        r_df_common.sort_index(inplace=True)

        if step == 'raw':
            current_directory_m = os.path.join(
                self.outputs_path, PREPROCESSED_SAMPLES, os.path.basename(os.path.dirname(m_file_path))
            )
            current_directory_r = os.path.join(
                self.outputs_path, PREPROCESSED_SAMPLES, os.path.basename(os.path.dirname(r_file_path))
            )
            os.makedirs(current_directory_m, exist_ok=True)
            os.makedirs(current_directory_r, exist_ok=True)
            self.path_to_microbiome = os.path.join(current_directory_m, microbiome_file_name)
            self.path_to_response = os.path.join(current_directory_r, response_file_name)
        else:
            current_directory_m = os.path.dirname(m_file_path)
            current_directory_r = os.path.dirname(r_file_path)
            response_file_name = response_file_name.replace(".tsv", ".parquet")

        with open(os.path.join(current_directory_m, microbiome_file_name), 'w') as f:
            m_df_common.to_csv(f, sep="\t")
        if step == 'raw':
            with open(os.path.join(current_directory_r, response_file_name), 'w') as f:
                r_df_common.to_csv(f, sep="\t")
        else:
            with open(os.path.join(current_directory_r, response_file_name), 'wb') as f:
                r_df_common.to_parquet(f, engine="fastparquet")
        message = (f"For {step} files, rows (samples) {non_common_indexes.values} were left out as appearing only in "
                   f"one of the files.")
        self.logger.info(MisMatchFiles(message))

    def prepare_input(self, file_type: str, file_path: Union[str, Path], threshold: float = 0.2) -> None:
        """
        This function prepares the microbiome file: sets index, checks for only numeric values,
        removes all rows with only zero values and all columns with less than the threshold share of non-zero values.

        :param: file_type: the type of the input; can take only "microbiome" or "response" values.
        :param: file_path: path to the input file.
        :param: threshold: the threshold for the number of non-zero values in a column allowed to keep it.
        :return: None, the function stores the edited file in a new directory.
        """
        self.logger.info(f"Running {self.prepare_input.__name__} function for the {file_type} file.")

        # check if the file_type is valid
        if file_type not in ["microbiome", "response"]:
            error_msg = f"File type `{file_type}` is not valid. Please use only 'microbiome' or 'response' values."
            raise ValueError(error_msg)

        # read the file
        df = pd.read_csv(file_path, sep="\t", index_col=0)

        # check if the content of the file is numbers only
        if not Preprocessor.is_numeric_dataframe(df):
            raise NonNumericDataFrameError

        if file_type == "microbiome":
            # Check for negative values
            has_negative_values = (df < 0).any().any()
            if has_negative_values:
                raise NegativeValuesDataFrameError

            # Keep only columns that have at least 2 non-zero values
            filtered_microbiome_df, removed_microbiome_cols = Preprocessor.remove_columns_with_less_than_two_positives(
                df
            )
            if removed_microbiome_cols:
                self.logger.info(
                    f'Columns {removed_microbiome_cols} have less than 2 positive values and were removed.'
                )
                df = filtered_microbiome_df.copy()
            # Keep only columns that have more than the threshold share of non-zero values
            filtered_microbiome_df, removed_microbiome_cols = Preprocessor.remove_columns_with_too_many_zeros(
                df, threshold
            )
            if removed_microbiome_cols:
                self.logger.info(
                    f'Columns {removed_microbiome_cols} have less '
                    f'than {threshold * 100}% non-zero values and were removed.')
                df = filtered_microbiome_df.copy()
            # remove all rows with only 0 values
            df = df.fillna(0).loc[(df > 0).any(axis=1)]

        else:
            # remove all cols with only 0 values
            df = df.fillna(0).loc[:, (df != 0).any(axis=0)]

            # TODO: REMOVE TEMPORARY CODE
            # select only every 100th column
            # df = df.iloc[:, ::100]

        # check if there is still a df remaining after the filtering process
        if any(size == 0 for size in df.shape):
            raise EmptyDataFrame

        if file_type == "microbiome":
            # check for a minimum of 3 samples and 3 taxa
            if any(size < 3 for size in df.shape):
                raise MinDataFrame(file_name=file_path)

            # Normalize each row by dividing by the sum of the row
            df = df.div(df.sum(axis=1), axis=0)
        else:
            # check for a minimum of 3 samples and 1 response variable
            if df.shape[0] < 3 or df.shape[1] < 1:
                raise MinDataFrame(file_name=file_path)

        if file_type == "microbiome" and self.imputation_flag:
            # Perform zero-replacement
            self.logger.info("ZERO REPLACEMENT ___\n")
            try:
                df = self.perform_cmultrepl_imputation(df)
            except ValueError as e:
                self.logger.error(f"Error during zero-replacement: {e}")
                sys.exit(1)
            self.logger.info("___ FINISHED ZERO REPLACEMENT\n")

        if file_type == "microbiome":
            directory_name = "microbiome"
            file_name = f"{os.path.split(file_path)[-1]}"
        else:
            directory_name = "response"
            file_name = f"{os.path.split(file_path)[-1]}".replace("_", "-")
        path_to_save = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, directory_name)
        os.makedirs(path_to_save, exist_ok=True)
        with open(os.path.join(path_to_save, file_name), 'w') as f:
            df.to_csv(f, sep="\t")
        self.logger.info(f"Saving files in: {path_to_save}")

    def perform_cmultrepl_imputation(
            self,
            raw_df: pd.DataFrame,
            label: Union[float, int, None] = 0, method: str = "GBM", adjust: bool = True,
            frac: float = 0.65
    ) -> pd.DataFrame:
        """Imputes missing or zero values in compositional data using a multiplicative replacement strategy and assigns
        it to the class, along with a dictionary of imputed samples. For details on the method see Palarea-Albaladejo J,
        Martín-Fernández JA (2015). zCompositions – R package for multivariate imputation of left-censored data under a
        compositional approach. Chemometrics and Intelligent Laboratory Systems, 143, 85–96.
        doi:10.1016/j.chemolab.2015.02.019; for documentation on `zCompositions` package see:
        https://cran.r-project.org/web/packages/zCompositions/zCompositions.pdf

        :param raw_df: The raw dataframe to perform imputation of missing values.
        :param label: Label for count zeros (default is 0).
        :param method: Bayesian multiplicative imputation: 'GBM' (geometric), 'SQ' (square root) or 'BL' (Bayes-Laplace)
        :param adjust: Replace imputed values with a fraction of minimum values in rows, if they turn to be above.
        :param frac: Fraction for minimum adjusted imputed value.
        :return: Modified df
        """
        # Check if the method is supported
        method = method.upper()
        if method not in ['GBM', 'SQ', 'BL']:
            error_msg = "Invalid method. Supported methods: 'GBM', 'SQ', 'BL'."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Preserve the original DataFrame
        x_df = raw_df.copy()

        # Label Handling
        if label is isinstance(label, float) or isinstance(label, int):
            if not np.any(x_df == label):
                self.logger.info(f"While running {self.perform_cmultrepl_imputation.__name__}, no {label} "
                                 f"values were found in the given data set")
            if label != 0 and np.any(x_df == 0):
                error_msg = "Zero values not labelled as count zeros were found in the data set"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if np.any(x_df.isna()):
                error_msg = "NaN values not labelled as count zeros were found in the data set"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            if np.any(x_df == 0):
                error_msg = "Zero values not labelled as count zeros were found in the data set"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if np.any(x_df.isna()):
                error_msg = "NaN values not labelled as count zeros were found in the data set"
                self.logger.info(error_msg)
                raise ValueError(error_msg)

        # Input data validation
        if np.any(x_df < 0):
            error_msg = f"{x_df.name} contains negative values"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(x_df, pd.DataFrame):
            error_msg = f"{x_df.name} must be a Pandas DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        x_df[x_df == label] = np.nan
        nan_values = np.isnan(x_df.values).any(axis=1)
        nan_indices = np.where(nan_values)[0]
        nan_row_names = [i for (i, flag) in zip(x_df.index.tolist(), nan_values) if flag]

        imputed_samples_dictionary = {}
        for idx, row_name in zip(nan_indices, nan_row_names):
            # Get the indices of the NaN values in the current row
            null_taxa = np.where(np.isnan(x_df.values[idx]))[0]
            # Store the indices in the dictionary
            imputed_samples_dictionary[row_name] = [
                col_name for (col, col_name) in enumerate(x_df.columns.tolist()) if col in null_taxa.tolist()
            ]
        self.imputed_samples_dict = invert_dict(imputed_samples_dictionary)

        # Move to NumPy arrays
        x_df = x_df.values
        N, D = x_df.shape
        s = D  # default for the `BL` method
        n = np.sum(x_df, axis=1, where=~np.isnan(x_df), dtype=float)

        # Calculate alpha based on x_df (excluding row i)
        alpha = np.zeros((N, D))

        for i in range(N):
            alpha[i, :] = np.nansum(x_df[np.arange(N) != i], axis=0)

        # Calculate t based on alpha
        t = alpha / np.nansum(alpha, axis=1, keepdims=True)

        # Check for GBM method and ensure there is enough information
        if method == "GBM" and np.any(t == 0):
            error_msg = "GBM method: not enough information to compute t hyper-parameter," \
                        "probably there are columns with < 2 positive values."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate s based on the selected method
        if method == "GBM":
            s = 1 / np.exp(np.nanmean(np.log(t), axis=1))
        elif method == "SQ":
            s = np.sqrt(n)

        # Reshape s and n to have a common size for broadcasting
        s_broadcastable = s.reshape(-1, 1)
        n_broadcastable = n.reshape(-1, 1)

        # Calculate the replacement matrix
        repl = t * (s_broadcastable / (n_broadcastable + s_broadcastable))

        modified_x_df = x_df.copy()  # Make a copy to avoid modifying the original array
        col_minimums = np.nanmin(x_df, axis=0)  # Get the minimum value for each column

        # Multiplicative Replacement on Closed Data
        for i in range(N):
            zero = np.isnan(x_df[i, :])
            modified_x_df[i, zero] = repl[i, zero]

            adjusted = 0
            # Check if any values need adjustment
            if adjust:
                adjust_mask = np.multiply(zero, (modified_x_df[i, :] > col_minimums))

                if np.any(adjust_mask):
                    f = np.where(adjust_mask)[0]
                    modified_x_df[i, f] = frac * col_minimums[f]
                    adjusted += len(f)

        # Normalize the output
        modified_row_sums = modified_x_df.sum(axis=1, keepdims=True)
        modified_x_df = modified_x_df / modified_row_sums.repeat(modified_x_df.shape[1], axis=1)
        modified_x_df = pd.DataFrame(modified_x_df, columns=raw_df.columns, index=raw_df.index.tolist())
        modified_x_df.index.name = 'SampleID'

        self.imputed_x = modified_x_df

        return modified_x_df

    def find_common_outlier_mask(self) -> pd.Series:
        """Runs through preprocessed responses and finds the common outlier mask if outlier removal is set to True.
        The common mask is used to filter both microbiome and response dataframes before proceeding to SparccKit"""

        self.logger.info("`OUTLIER_REMOVAL` is set to True, preparing outlier mask.")
        if (
            os.path.exists(self.meta_data)
            and "Category" in pd.read_csv(self.meta_data, sep="\t", index_col=0).columns
        ):
            self.logger.info("Meta data file detected with 'Category' column in it.")
        else:
            self.logger.info("No meta data file detected or no 'Category' column in the meta data file.")

        masked_responses_file_name = os.path.join(
            os.path.join(self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION),
            "masked_" + os.path.basename(self.path_to_prepared_response)
        )
        response_p_values_file_name = os.path.join(
            os.path.join(self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION),
            "bonferroni_p_values_" + os.path.basename(self.path_to_prepared_response)
        )

        response_df = pd.read_csv(self.path_to_prepared_response, sep="\t", index_col=0)
        if response_df.shape[1] > 1:
            bonferroni_correction = response_df.shape[1]
            self.logger.info("Multiple response variables detected; looking for common samples.")
            masked_responses_df = response_df.copy()
            response_p_vals_df = response_df.copy()
            # Run through every column and change it to its mask from prepare_response_outlier_mask function
            for col_num, col in enumerate(response_df.columns):
                response_series = response_df[col]
                mask, p_vals = self.prepare_response_outlier_mask(response_series, col_num, bonferroni_correction)
                masked_responses_df[col] = mask
                response_p_vals_df[col] = p_vals

            with open(masked_responses_file_name, 'w') as f:
                masked_responses_df.to_csv(f, sep="\t")
            # Find a common mask
            with open(response_p_values_file_name, 'w') as f:
                response_p_vals_df.to_csv(f, sep="\t")

            common_mask = masked_responses_df.all(axis=1)
            self.logger.info(f"Common mask found with {common_mask.sum()} samples remaining out of "
                             f"{len(common_mask)} total samples. Please check the outlier stats file in the "
                             f"`{PREPROCESSING_RESULTS}/{OUTLIER_DETECTION}` folder for details.")
        else:
            self.logger.info("Single response variable detected; looking for outliers.")
            response_series = response_df.iloc[:, 0]
            common_mask, p_vals = self.prepare_response_outlier_mask(response_series)

            with open(masked_responses_file_name, 'w') as f:
                common_mask.to_csv(f, sep="\t")
            # Find a common mask
            with open(response_p_values_file_name, 'w') as f:
                p_vals.to_csv(f, sep="\t")

            self.logger.info(f"Mask found with {common_mask.sum()} samples remaining out of "
                             f"{len(common_mask)} total samples.")
        common_mask.name = 'CommonMask'

        return common_mask.astype(bool)

    @staticmethod
    def _compute_outlier_mask_and_p_values(series: pd.Series, alpha: float) -> tuple[
        pd.Series, pd.Series, floating[Any] | float, float, float, float
    ]:
        """Compute robust outlier mask and p-values for a numeric series."""
        # Ensure numeric and drop NaNs for robust statistics
        series_numeric = pd.to_numeric(series, errors='coerce').dropna()
        if series_numeric.empty:
            # No valid data -> return all-non-outlier mask and neutral p-values
            mask = pd.Series(True, index=series.index, dtype=bool)
            p_vals = pd.Series(1.0, index=series.index, dtype=float)
            return mask, p_vals, float('nan'), float('nan'), float('nan'), float('nan')

        median = np.median(series_numeric)
        std = robust.mad(series_numeric)

        # Fallback to sample std if MAD is zero or invalid
        if not np.isfinite(std) or std <= 0:
            if series_numeric.size > 1:
                std = np.std(series_numeric, ddof=1)
            else:
                std = 0.0

        # Final safe fallback to a small positive epsilon to avoid zero/NaN scale
        eps = 1e-8
        if not np.isfinite(std) or std <= 0:
            std = eps

        lower = norm.ppf(alpha, loc=median, scale=std)
        upper = norm.ppf(1 - alpha, loc=median, scale=std)

        # Compute z-scores safely using the same median/std; preserve original index and NaNs
        z_vals = (series.to_numpy(dtype=float) - median) / std
        z = pd.Series(z_vals, index=series.index, dtype=float)
        p_vals = pd.Series(2 * norm.sf(np.abs(z)), index=series.index, dtype=float)
        mask = pd.Series((series >= lower) & (series <= upper), index=series.index, dtype=bool)

        return mask, p_vals, median, std, lower, upper

    def prepare_response_outlier_mask(
            self, response_series: pd.Series, response_index: int = 0, correction_coef: float = None
    ) -> tuple[pd.Series, pd.Series]:
        """Computes p-values and a boolean mask by response variable. Uses median and MAD as robust estimators of
        location and scale, respectively. If categorical metadata is provided, outlier detection is performed within
        each category. Returns a boolean mask where True indicates a non-outlier sample, and per-sample p-values.

        :param response_series: A Pandas Series representing the response variable.
        :param response_index: An integer representing the index of the response variable (default is 0).
        :param correction_coef: A float representing the correction coefficient. If None, no correction is applied.
        :return: A boolean Series where True indicates an outlier.
        """

        outlier_alpha_per_response = OUTLIER_ALPHA / correction_coef if correction_coef else OUTLIER_ALPHA
        mask = pd.Series(False, index=response_series.index, name=response_series.name)
        p_vals = pd.Series(1.0, index=response_series.index, name=response_series.name)

        if (
            os.path.exists(self.meta_data)
            and "Category" in pd.read_csv(self.meta_data, sep="\t", index_col=0).columns
        ):
            df_meta = pd.read_csv(self.meta_data, sep="\t", index_col=0)["Category"].copy()

            # Make sure df_meta only contains indices present in response_series
            df_meta = df_meta[df_meta.index.isin(response_series.index)]

            groups = df_meta.dropna().unique()
            # Compute outliers per group
            for g in groups:
                idx = df_meta == g
                # Check if the group has enough samples to compute outliers
                if idx.sum() < MIN_SAMPLES_FOR_OUTLIER_REMOVAL:
                    mask.loc[idx.index[idx]] = True
                    mask = pd.Series(mask, index=response_series.index, dtype=bool)
                    self.logger.info(
                        f"For response '{response_series.name}', "
                        f"group '{g}' has less than {MIN_SAMPLES_FOR_OUTLIER_REMOVAL} points; skipping this group."
                    )
                    continue

                else:
                    y_group = response_series.loc[idx.index[idx]]
                    mask_group, pvals_group, median, std, lower, upper = self._compute_outlier_mask_and_p_values(
                        y_group,
                        outlier_alpha_per_response,
                    )

                    mask.loc[idx.index[idx]] = mask_group
                    p_vals.loc[idx.index[idx]] = pvals_group
                    # Check that the masked group has not less than MIN_SAMPLES_FOR_OUTLIER_REMOVAL points
                    if mask.loc[idx.index[idx]].sum() < MIN_SAMPLES_FOR_OUTLIER_REMOVAL:
                        mask.loc[idx.index[idx]] = True
                        mask = pd.Series(mask, index=response_series.index, dtype=bool)
                        self.logger.info(
                            f"For response '{response_series.name}', outlier removal will leave less than "
                            f"{MIN_SAMPLES_FOR_OUTLIER_REMOVAL} points in group '{g}'; skipping the case."
                        )
                    else:
                        self._update_mask_stats_file(
                            response_series, response_index, mask, median, std, lower, upper, group=g, group_index=idx
                        )

        else:
            # If no groups detected, mask outliers in the entire response series
            if len(response_series) < MIN_SAMPLES_FOR_OUTLIER_REMOVAL:
                mask[:] = True
                self.logger.info(
                    f"Total points less than {MIN_SAMPLES_FOR_OUTLIER_REMOVAL}; skipping outlier removal."
                )
            else:
                mask, p_vals, median, std, lower, upper = self._compute_outlier_mask_and_p_values(
                    response_series,
                    outlier_alpha_per_response,
                )
                self._update_mask_stats_file(response_series, response_index, mask, median, std, lower, upper)

        if correction_coef:
            p_vals = (p_vals * correction_coef).clip(upper=1)
        return mask.astype(bool), p_vals

    def _update_mask_stats_file(
            self, response_series, response_index, mask, median, std, lower, upper, group=None, group_index=None
    ):
        stats_file_path = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION)
        os.makedirs(stats_file_path, exist_ok=True)
        stats_file_name = os.path.join(stats_file_path, "outlier_stats.tsv")
        write_header = not os.path.exists(stats_file_name)
        with open(stats_file_name, "a") as stats_file:
            if write_header:
                stats_file.write("Response_Tag\tResponse_Median\tResponse_STD\tLower\tUpper\tOutlier_Sample(s)\n")
            r = f"response_{response_index + 1}_{response_series.name}"
            m_f = mask
            if group_index is not None:
                r = f"{r}_{group}"
                m_f = mask & group_index
            stats_file.write(f"{r}\t{median:.4f}\t{std:.4f}\t{lower:.4f}\t{upper:.4f}\t{mask[~ m_f].index.tolist()}\n")

    def calculate_slopes_and_ses(
            self, combination, microbiome_df, response_df,
            outlier_detection_flag: bool, all_samples_flag: bool, masked_samples_flag: bool,
            n_taxa_dictionary: dict = None
    ):
        self.logger.debug(f"Response index value: {combination.response_index}")
        compores_basic_run_result = combination.run_comp_process_task(
            microbiome_df, microbiome_df.shape[1], self.file_name, response_df,
            outlier_detection_step_flag=outlier_detection_flag,
            outlier_detection_all=all_samples_flag,
            outlier_detection_masked=masked_samples_flag,
            n_taxa=n_taxa_dictionary
        )
        slope_array, se_array, n_taxa_dict = (
            compores_basic_run_result[3],
            compores_basic_run_result[6],
            compores_basic_run_result[5][f"{microbiome_df.shape[1]} OCUs"]['NUM_OCU'],
        )
        self.logger.debug(f"Nominator taxa dictionary: {n_taxa_dict}")

        return slope_array, se_array, n_taxa_dict

    def calculate_slope_change_p_values(
            self,
            slope_array_before: np.ndarray, se_array_before: np.ndarray,
            slope_array_after: np.ndarray, se_array_after: np.ndarray,
            response_index: list
    ) -> np.ndarray:
        """Compares two arrays of slopes and their standard errors to determine if they are significantly different.
        :param slope_array_before: Array of slopes before filtering.
        :param se_array_before: Array of standard errors before filtering.
        :param slope_array_after: Array of slopes after filtering.
        :param se_array_after: Array of standard errors after filtering.
        :param response_index: List of response variable tags corresponding to the slopes.
        :return: A boolean array indicating whether each pair of slopes is significantly different.
        """
        if not (len(slope_array_before) == len(se_array_before) == len(slope_array_after) == len(se_array_after)):
            error_msg = "All input arrays must have the same length."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        slope_diff = abs(slope_array_before - slope_array_after)
        slope_sigma = np.sqrt(se_array_before**2 + se_array_after**2)
        slope_z_scores = slope_diff / slope_sigma
        slope_p_values_before_correction = norm.sf(np.abs(slope_z_scores))

        self._update_slope_change_stats_file(
            slope_array_before, se_array_before, slope_array_after, se_array_after,
            slope_diff, slope_sigma, slope_z_scores, slope_p_values_before_correction, response_index
        )
        return slope_p_values_before_correction

    def _update_slope_change_stats_file(
            self, slopes_before: np.ndarray, ses_before: np.ndarray, slopes_after: np.ndarray, ses_after: np.ndarray,
            slope_deltas: np.ndarray, slope_delta_sigmas: np.ndarray, z_scores: np.ndarray, raw_p_values: np.ndarray,
            response_tags: list
    ):
        stats_file_path = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, OUTLIER_DETECTION)
        os.makedirs(stats_file_path, exist_ok=True)
        stats_file_name = os.path.join(stats_file_path, "slope_change_stats.tsv")
        write_header = not os.path.exists(stats_file_name)
        with open(stats_file_name, "a") as stats_file:
            if write_header:
                stats_file.write(
                    "Response Tag\tSlope_Before\tSE_Before\tSlope_After\tSE_After\t"
                    "Slope_Diff\tSlope_Diff_Sigma\tZ_score\tBonferroni_P_value\n")
            for sb, seb, sa, sea, sd, ss, z, p, response_tag in zip(
                    slopes_before, ses_before, slopes_after, ses_after,
                    slope_deltas, slope_delta_sigmas, z_scores, raw_p_values, response_tags
            ):
                p = min(p * len(slopes_before), 1.0)  # Bonferroni correction
                stats_file.write(
                    f"{response_tag}\t{sb:.6f}\t{seb:.6f}\t{sa:.6f}\t{sea:.6f}\t{sd:.6f}\t{ss:.6f}\t{z:.6f}\t{p:.6f}\n")

    def create_linkage(self) -> None:
        """
        Creates linkage matrix according to correlation between microbe genomes. Takes a path to the correlation file
        (assumed to be a tab-separated file with microbe correlations) and returns hierarchical clustering encoded
        as a linkage matrix.

        Note: SciPy's hierarchical clustering (linkage function) expects a condensed distance matrix as an intput: if
        not, a 2-D array is treated as a collection of observation vectors to be clustered, and the condensed distance
        matrix is evaluated during the run, using the `pdist` function and the `metric` parameter value, `euclidean` by
        default. Thus, the `squareform` function is used to convert a square distance matrix into a condensed form to
        pass it to the linkage function.

        Note: Since distances are derived from correlations, the "average" linkage method is applied as more suitable.
        The "average" linkage method calculates the average distance between all pairs of points in two clusters,
        making it less sensitive to outliers.
        """
        corr_file = self.path_to_sparcckit_corr
        # Read the correlation file
        corr = pd.read_csv(corr_file, sep="\t", index_col=0)
        # Calculate the distance matrix
        distance_matrix = .5 * (1 - corr)
        # Convert the distance matrix to condensed form
        condensed_distance_matrix = squareform(distance_matrix)
        # Create the linkage matrix
        linkage_matrix = linkage(condensed_distance_matrix, method='average')
        self.linkage_matrix = linkage_matrix
        message = "Linkage matrix generated."
        self.logger.info(message)

    def save_ocu_taxa_map_csv(self, file_name: str, cluster_count: int, clustering_dictionary: dict[str:[]]) -> None:
        """Writes the mapping between OCU and the OTU it contains to a `csv` file in corresponding balance plot folders.

        :param file_name: Name of the experiment
        :param cluster_count: Current number of OCU
        :param clustering_dictionary:
        :return: None
        """
        for method in self.balance_methods:
            # Create a directory for the response vs. balance plots if it doesn't exist
            directory_path = os.path.join(self.path_response_vs_balance_plots, method, str(cluster_count))
            os.makedirs(directory_path, exist_ok=True)
            # Save OCU-to-taxa mapping as CSV
            mapping_csv_path = os.path.join(directory_path, f"{file_name}_{cluster_count}_OCU_taxa_mapping.csv")
            with open(mapping_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['OCU', 'Taxa'])
                for ocu_name, data in clustering_dictionary[f"{cluster_count} OCUs"]["OCUs"].items():
                    taxa_list = ";".join(data['taxa'])  # Join taxa with semicolon or any delimiter you prefer
                    writer.writerow([ocu_name, taxa_list])

    def save_ocu_matrices(self) -> None:
        """The function takes the output of the `create_linkage` function (linkage matrix), forms clusters from the
        hierarchical clustering defined by the given linkage matrix, fetches corresponding OTUs (columns) of the initial
        preprocessed microbiome matrix, and groups the OTU values into OCUs. Resulting OCU matrices and clustering
        metadata are stored in a subdirectory of the output directory defined in the config file (takes the path
        to the preprocessed microbiome file and the path to the output directory to store the output).
        """
        file_name = self.file_name
        preprocessed_microbiome_file_path = self.path_to_microbiome_clustering
        clustered_ocu_path = os.path.join(self.clustered_ocu_path, self.file_name)

        os.makedirs(clustered_ocu_path, exist_ok=True)

        df = pd.read_csv(preprocessed_microbiome_file_path, sep="\t", index_col=0)
        column_labels = df.columns

        # Initialize a dictionary to store the clustering metadata with the original OTU case
        clustering_dictionary = {
            f'{len(column_labels)} OCUs': {'threshold': 0, 'OCUs': {}}
        }
        for i, l in enumerate(column_labels):
            clustering_dictionary[f'{len(column_labels)} OCUs']['OCUs'][f'ocu_{i + 1}'] = {}
            clustering_dictionary[f'{len(column_labels)} OCUs']['OCUs'][f'ocu_{i + 1}']['taxa'] = [l]

            if l in self.imputed_samples_dict:
                clustering_dictionary[f'{len(column_labels)} OCUs']['OCUs'][f'ocu_{i + 1}']['imputed_in'] = (
                    self.imputed_samples_dict[l]
                )
            else:
                clustering_dictionary[f'{len(column_labels)} OCUs']['OCUs'][f'ocu_{i + 1}']['imputed_in'] = []

        # Fetch thresholds from the linkage matrix, group clusters, and sum OTU values for members in each cluster
        for i, t in enumerate(self.linkage_matrix[:, 2]):
            # Sample clustering cases with a step defined by OCU_SAMPLING_RATE to dilute further processing
            # Start from a total number of given valid OTUs and stop at MIN_OCU_NUM
            if (i + 1) % self.ocu_sampling_rate != 0:
                continue

            clusters = fcluster(self.linkage_matrix, t, criterion='distance')
            cluster_count = len(np.unique(clusters))
            if cluster_count < MIN_OCU_NUM:
                break

            # Specify the directory path for the current threshold
            path = os.path.join(clustered_ocu_path, str(cluster_count))
            # Specify the file path for the OCU matrix
            ocu_matrix_file_name = f"{file_name}_{cluster_count}_OCUs.tsv"
            ocu_matrix_path = os.path.join(path, ocu_matrix_file_name)

            d_key = f'{cluster_count} OCUs'

            # Perform clustering only if the number of clusters differs from the previous case
            if not os.path.exists(path) or os.listdir(path) == []:
                os.makedirs(path, exist_ok=True)
                clustering_dictionary[d_key] = {'threshold': t, 'OCUs': {}}

                # Create a new DataFrame to write clustered OTU columns
                clustered_columns_df = pd.DataFrame(index=df.index)

                for cluster_label in np.unique(clusters):
                    # Extract columns belonging to the current cluster
                    cluster_columns = column_labels[clusters == cluster_label]

                    # Create a new column in the DataFrame with the sum of values for the cluster
                    new_column = df[cluster_columns].sum(axis=1).rename(f'ocu_{cluster_label}')
                    clustered_columns_df = pd.concat([clustered_columns_df, new_column], axis=1)
                    clustering_dictionary[d_key]['OCUs'][f'ocu_{cluster_label}'] = {}
                    clustering_dictionary[d_key]['OCUs'][f'ocu_{cluster_label}']['taxa'] = cluster_columns.tolist()
                    clustering_dictionary[d_key]['OCUs'][f'ocu_{cluster_label}']['imputed_in'] = []

                    imputed_in_list = []
                    for col in cluster_columns:
                        if col in self.imputed_samples_dict:
                            imputed_in_list += self.imputed_samples_dict[col]
                    imputed_in_list = list(set(imputed_in_list))
                    clustering_dictionary[d_key]['OCUs'][f'ocu_{cluster_label}']['imputed_in'] = imputed_in_list

                # Write the resulting DataFrame to a CSV file
                with open(ocu_matrix_path, 'w') as f:
                    clustered_columns_df.to_csv(f, sep="\t")

                self.save_ocu_taxa_map_csv(file_name, cluster_count, clustering_dictionary)

        self.clustered_ocu_dictionary = clustering_dictionary

        non_clustered_ocu_matrix_path = os.path.join(clustered_ocu_path, str(len(column_labels)))
        column_labels = [f'ocu_{i + 1}' for i in range(len(column_labels))]
        non_clustered_ocu_df = df.copy()
        non_clustered_ocu_df.columns = column_labels
        os.makedirs(non_clustered_ocu_matrix_path, exist_ok=True)
        non_clustered_ocu_matrix_file_name = (
                non_clustered_ocu_matrix_path + f"/{file_name}_{len(column_labels)}_OCUs.tsv"
        )
        with open(non_clustered_ocu_matrix_file_name, 'w') as f:
            non_clustered_ocu_df.to_csv(f, sep="\t")

        self.save_ocu_taxa_map_csv(file_name, len(column_labels), clustering_dictionary)

        # Specify the file path for the OCU dictionary JSON
        ocu_json_file_path = os.path.join(clustered_ocu_path, f"{file_name}_ocu_clustering_dictionary.json")
        # Specify the file path for the imputed sample dictionary JSON
        imputed_json_file_path = os.path.join(clustered_ocu_path, f"{file_name}_imputed_samples_dictionary.json")

        # Write the OCU dictionary to a JSON file
        with open(ocu_json_file_path, 'w') as json_file:
            json.dump(self.clustered_ocu_dictionary, json_file, indent=4)

        # Write the OCU dictionary to a JSON file
        with open(imputed_json_file_path, 'w') as json_file:
            json.dump(self.imputed_samples_dict, json_file, indent=4)

        message = "OCU clustering generated."
        self.logger.info(message)

    @staticmethod
    def is_numeric_dataframe(df):
        # Convert values to numeric types, coercing non-numeric values to NaN
        numeric_data = df.apply(pd.to_numeric, errors='coerce')

        # Check for the appearance of NaN values
        non_numeric_mask = pd.isna(numeric_data)

        # Check if all values are numeric
        return not non_numeric_mask.any().any()

    @staticmethod
    def remove_columns_with_less_than_two_positives(df):
        # keeps only columns with more than 2 positive values
        positive_counts = (df > 0).sum()
        filtered_df = df.loc[:, positive_counts >= 2]
        removed_columns = df.columns.difference(filtered_df.columns).tolist()
        return filtered_df, removed_columns

    @staticmethod
    def remove_columns_with_too_many_zeros(df, threshold=0.2):
        # keeps only columns with more than 2 positive values
        mean_positive_counts = (df > 0).mean()
        filtered_df = df.loc[:, mean_positive_counts > threshold]
        removed_columns = df.columns.difference(filtered_df.columns).tolist()
        return filtered_df, removed_columns
