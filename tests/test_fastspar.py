import os.path
import subprocess
import numpy as np
import pandas as pd
import pytest


# TODO: Extract a separate run_fastspar function from run_process_one_case_combination?
class TestCallFastSpar:

    @pytest.fixture(scope="function")
    def setup_teardown_fastspar(self, tmp_path, num_individuals=500, normalization_constant=10000):
        """
        [1] FastSpar: rapid and scalable correlation estimation for compositional data,  29 August 2018,
        Stephen C Watts, Scott C Ritchie, Michael Inouye, Kathryn E Holt, https://doi.org/10.1093/bioinformatics/bty734;

        [2] Inferring Correlation Networks from Genomic Survey Data, Jonathan Friedman,Eric J. Alm, September 20, 2012,
        https://doi.org/10.1371/journal.pcbi.1002687,
        https://github.com/Zhenev/sparcc/blob/master/sparcc/simulate_data.py
        """

        # Define the path parameters for calling fastspar
        path_to_input_data = tmp_path / "test_community_matrix.tsv"
        path_to_save_fastspar_cov_file = tmp_path / "test_fastspar_cov_file.tsv"
        path_to_save_fastspar_cor_file = tmp_path / "test_fastspar_cor_file.tsv"

        # Generate synthetic data of the required form and store it in path_to_input_data
        cov_matrix = pd.read_csv(f"{os.getcwd()}/tests/data/fastspar_fake_data_cov.tsv", sep="\t", index_col=0)  # [1]
        cov_matrix_length = cov_matrix.shape[0]
        otu_0 = 0.1  # [2]
        otu_1 = 0.9 * otu_0
        mean_vector = np.array([otu_1]+[(otu_0-otu_1)/(cov_matrix_length-1)]*(cov_matrix_length-1))
        samples = np.exp(np.random.multivariate_normal(mean_vector, cov_matrix, num_individuals))
        counts = samples * normalization_constant
        rounded_samples = counts.round()
        df = pd.DataFrame(rounded_samples)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.T
        df.index = cov_matrix.index
        df.index.name = "#OTU ID"
        df.to_csv(path_to_input_data, sep="\t")

        # Define the path parameters to the expected output
        path_to_expected_cov_file = os.path.join(os.getcwd(), "tests/data/fastspar_fake_data_cov.tsv")
        path_to_expected_cor_file = os.path.join(os.getcwd(), "tests/data/fastspar_fake_data_cor.tsv")

        yield path_to_input_data, path_to_save_fastspar_cov_file, path_to_save_fastspar_cor_file, \
            path_to_expected_cov_file, path_to_expected_cor_file

    def test_call_fastspar(self, setup_teardown_fastspar):

        # Call the fixture to get the path parameters for calling fastspar and the expected output
        input_path, output_cov_matrix_path, output_cor_matrix_path, exp_cov_path, exp_cor_path = setup_teardown_fastspar

        # Run fastspar command
        command = [
            'fastspar',
            '--iterations', '100',
            '--otu_table', input_path,
            '--correlation', output_cor_matrix_path,
            '--covariance', output_cov_matrix_path
        ]
        subprocess.run(command, check=True)

        # Read the resulting and expected matrices
        original_cov = pd.read_csv(exp_cov_path, sep="\t")
        results_cov = pd.read_csv(output_cov_matrix_path, sep="\t")
        original_cor = pd.read_csv(exp_cor_path, sep="\t")
        results_cor = pd.read_csv(output_cor_matrix_path, sep="\t")

        # Define parameters for evaluating the resulting matrices against the expected ones
        relative_tolerance = 0.01
        absolute_tolerance = 0.1
        dim = original_cov.shape[0]
        dim = dim * (dim - 1)

        # Assert the existence of the output files
        assert os.path.exists(output_cov_matrix_path)
        assert os.path.exists(output_cor_matrix_path)
        # Assert similarity of the resulting and expected matrices
        assert np.sum(np.isclose(results_cor, original_cor, rtol=relative_tolerance, atol=absolute_tolerance)) > dim
        assert np.sum(np.isclose(results_cov, original_cov, rtol=relative_tolerance, atol=absolute_tolerance)) > dim

