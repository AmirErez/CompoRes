import json
import os
import sys
from logging import Logger
import subprocess
import csv
from pathlib import Path
from typing import Union, AnyStr

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .exceptions_module import MisMatchFiles, NonNumericDataFrameError, NegativeValuesDataFrameError, \
    EmptyDataFrame, MinDataFrame
from .utils import invert_dict

PREPROCESSING_RESULTS = 'preprocessing_results'

MIN_OCU_NUM = 3


class Preprocessor:

    def __init__(
            self,
            logger: Logger,
            s1: str, s2: str, s3: str,
            path_to_microbiome: Union[Path, str, AnyStr],
            path_to_response: Union[Path, str, AnyStr],
            path_to_microbiome_clustering: Union[Path, str, AnyStr],
            path_to_prepared_response: Union[Path, str, AnyStr],
            path_to_fastspar_results: Union[Path, str, AnyStr],
            path_to_fastspar_corr: Union[Path, str, AnyStr],
            path_to_fastspar_cov: Union[Path, str, AnyStr],
            path_to_outputs: Union[Path, str, AnyStr],
            path_to_clustered_ocu: Union[Path, str, AnyStr],
            path_to_response_vs_balance_plots: Union[Path, str, AnyStr],
            clustering_sampling_rate: int,
            imputation_flag: bool = True
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
        self.path_to_fastspar_results = path_to_fastspar_results
        self.path_to_fastspar_input = os.path.join(self.outputs_path, PREPROCESSING_RESULTS, 'fastspar')
        self.path_to_fastspar_corr = path_to_fastspar_corr
        self.path_to_fastspar_cov = path_to_fastspar_cov
        self.clustered_ocu_dictionary = {}
        self.file_name = f"{self.s1}-{self.s2}-{self.s3}"
        self.imputation_flag = imputation_flag
        self.path_response_vs_balance_plots = path_to_response_vs_balance_plots
        self.ocu_sampling_rate = clustering_sampling_rate

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
            self.check_input_files_for_same_indexes(self.path_to_microbiome, self.path_to_response, step='raw')
        except (MisMatchFiles, MinDataFrame) as e:
            self.logger.error(f"Error checking input files for same indexes: {e}")
            sys.exit(1)

        try:
            self.prepare_input("microbiome", self.path_to_microbiome)
            self.prepare_input("response", self.path_to_response)
        except (NonNumericDataFrameError, NegativeValuesDataFrameError, EmptyDataFrame, MinDataFrame, ValueError) as e:
            self.logger.error(f"Error preparing input: {e}")
            sys.exit(1)

        try:
            self.check_input_files_for_same_indexes(
                self.path_to_microbiome_clustering, self.path_to_prepared_response, step='preprocessed'
            )
        except (MisMatchFiles, MinDataFrame) as e:
            self.logger.error(f"Error checking input files for same indexes: {e}")
            sys.exit(1)

        self.fastspar_prepare()
        os.makedirs(self.path_to_fastspar_results, exist_ok=True)
        command = [
            'fastspar',
            '--iterations', '100',
            '--otu_table', (os.path.join(self.path_to_fastspar_input, f"{self.file_name}.tsv")),
            '--correlation',
            self.path_to_fastspar_corr,
            '--covariance',
            self.path_to_fastspar_cov
        ]
        subprocess.run(command)
        message = "Fastspar command has been run."
        self.logger.info(message)

        self.create_linkage()
        self.save_ocu_matrices()

        self.logger.info("Preprocessing finished.")

    def check_input_files_for_same_indexes(
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

        non_common_indexes = m_df_raw.index.symmetric_difference(r_df_raw.index)
        common_indexes = m_df_raw.index.intersection(r_df_raw.index)
        m_df_common = m_df_raw.loc[common_indexes].copy()
        r_df_common = r_df_raw.loc[common_indexes].copy()

        microbiome_file_name = os.path.basename(m_file_path)
        response_file_name = os.path.basename(r_file_path)

        # check for a minimum of 3 samples and 3 taxa
        if any(size < 3 for size in m_df_common.shape):
            self.logger.exception(MinDataFrame(file_name=microbiome_file_name))
            raise MinDataFrame(file_name=microbiome_file_name)
        if r_df_common.shape[0] < 3 or r_df_common.shape[1] < 1:
            self.logger.exception(MinDataFrame(file_name=response_file_name))
            raise MinDataFrame(file_name=response_file_name)

        m_df_common.sort_index(inplace=True)
        r_df_common.sort_index(inplace=True)

        if step == 'raw':
            current_directory_m = os.path.join(
                self.outputs_path, 'preprocessed_samples', os.path.basename(os.path.dirname(m_file_path))
            )
            current_directory_r = os.path.join(
                self.outputs_path, 'preprocessed_samples', os.path.basename(os.path.dirname(r_file_path))
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
        message = f"""For {step} files, rows (samples) {non_common_indexes.values} were left out as appearing only
        in one of the files."""
        self.logger.info(MisMatchFiles(message))

    def prepare_input(self, file_type: str, file_path: Union[str, Path], threshold: float = 0.2) -> None:
        """
        This function prepares the microbiome file: sets index, checks for only numeric values,
        removes all rows with only zero values and all columns with less than threshold share of non-zero values.

        :param: file_type: the type of the input; can take only "microbiome" or "response" values.
        :param: file_path: path to the input file.
        :param: threshold: the threshold for the number of non-zero values in a column allowed to keep it.
        :return: None, the function stores the edited file in a new directory.
        """
        self.logger.info(f"Running {self.prepare_input.__name__} function for the {file_type} file.")

        # check if the file_type is valid
        if file_type not in ["microbiome", "response"]:
            error_msg = f"File type `{file_type}` is not valid. Please use only 'microbiome' or 'response' values."
            self.logger.exception(ValueError(error_msg))
            raise ValueError(error_msg)

        # read the file
        df = pd.read_csv(file_path, sep="\t")
        df = df.set_index(df.columns[0])

        # check if the content of the file is numbers only
        if not Preprocessor.is_numeric_dataframe(df):
            self.logger.exception(NonNumericDataFrameError)
            raise NonNumericDataFrameError

        if file_type == "microbiome":
            # Check for negative values
            has_negative_values = (df < 0).any().any()
            if has_negative_values:
                self.logger.exception(NegativeValuesDataFrameError)
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
            # Keep only columns that have more than threshold share of non-zero values
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

        # check if there is still a df remaining after the filtering process
        if any(size == 0 for size in df.shape):
            self.logger.exception(EmptyDataFrame)
            raise EmptyDataFrame

        if file_type == "microbiome":
            # check for a minimum of 3 samples and 3 taxa
            if any(size < 3 for size in df.shape):
                self.logger.exception(MinDataFrame(file_name=file_path))
                raise MinDataFrame(file_name=file_path)
            # Normalize each row by dividing by the sum of the row
            df = df.div(df.sum(axis=1), axis=0)
        else:
            # check for a minimum of 3 samples and 1 response variable
            if df.shape[0] < 3 or df.shape[1] < 1:
                self.logger.exception(MinDataFrame(file_name=file_path))
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

    def fastspar_prepare(self) -> None:
        """Prepares the microbiome file to run in Fastspar: removes all columns with only 0 values; changes the index
         name to #OTU ID; then, stores the result.
        """
        microbiome_file = self.path_to_microbiome_clustering
        df = pd.read_csv(microbiome_file, sep="\t", index_col=0)
        # remove all columns with only 0 values
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.T
        # change the index name to #OTU ID
        df.index.name = "#OTU ID"

        # Store the reformatted file in the Fastspar input directory
        os.makedirs(self.path_to_fastspar_input, exist_ok=True)
        with open(os.path.join(self.path_to_fastspar_input, f"{self.file_name}.tsv"), 'w') as f:
            df.to_csv(f, sep="\t")
        message = "The preprocessed microbiome file has been reformatted to run in Fastspar."
        self.logger.info(message)

    def create_linkage(self) -> None:
        """
        Creates linkage matrix according to correlation between microbe genomes. Takes a path to the correlation file
        (assumed to be a tab-separated file with microbe correlations) and returns hierarchical clustering encoded
        as a linkage matrix.

        Note: SciPy's hierarchical clustering (linkage function) expects a condensed distance matrix as an intput:
        a 2-D array is treated as a collection of observation vectors to be clustered, and the condensed distance matrix
        is evaluated during the run, using the `pdist` function and the `metric` parameter value, `euclidian` by
        default. The `squareform` function is used to convert a square distance matrix into a condensed form to pass it
        to the linkage function.

        Note: Since distances are derived from correlations, the "average" linkage method is applied as more suitable.
        The "average" linkage method calculates the average distance between all pairs of points in two clusters,
        making it less sensitive to outliers.
        """
        corr_file = self.path_to_fastspar_corr
        # Read the correlation file
        corr = pd.read_csv(corr_file, sep="\t", index_col=0)
        # Calculate the distance matrix
        distance_matrix = .5 * np.sqrt(1 - np.square(corr))
        # Convert the distance matrix to condensed form
        condensed_distance_matrix = squareform(distance_matrix)
        # Create the linkage matrix
        linkage_matrix = linkage(condensed_distance_matrix, method='average')
        self.linkage_matrix = linkage_matrix
        message = "Linkage matrix generated."
        self.logger.info(message)

    def save_ocu_taxa_map_csv(self, file_name: str, cluster_count: int, clustering_dictionary: dict[str:[]],
                              d_key: str) -> None:
        """Keeps csv file mapping between ocu and the taxa it contains in the plot directory, under CLR
        and the total ocu number of the current ocu cluster.
        :param file_name: Name of the experiment
        :param cluster_count: Current ocu number
        :param clustering_dictionary:
        :param d_key: Current ocu number key name
        :return: None`
        """
        directory_path = os.path.join(self.path_response_vs_balance_plots, str(cluster_count))
        os.makedirs(directory_path, exist_ok=True)
        # Save OCU-to-taxa mapping as CSV
        mapping_csv_path = os.path.join(directory_path, f"{file_name}_{cluster_count}_OCU_taxa_mapping.csv")
        with open(mapping_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['OCU', 'Taxa'])
            for ocu_name, data in clustering_dictionary[d_key]['OCUs'].items():
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

                self.save_ocu_taxa_map_csv(file_name, cluster_count, clustering_dictionary, d_key)

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
