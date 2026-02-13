import logging
import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from src.compores.sparcckit import SparccKit

import matplotlib.pyplot as plt

from utils.generate_synthetic_data import set_up_compositional_data_paths, simulate_microbiome_data


class TestSparccKit:
    @pytest.fixture(scope="function")
    def logger(self):
        logger = logging.getLogger(__name__)

        return logger

    @pytest.fixture
    def setup_teardown_fractions(self):
        np.random.seed(42)
        data = np.random.dirichlet(alpha=[1.0] * 3, size=100)  # 100 samples, 5 components
        yield pd.DataFrame(data, columns=['A', 'B', 'C'])

    def test_compute_variation_matrix(self, setup_teardown_fractions, logger):
        fractions = setup_teardown_fractions
        model = SparccKit(
            logger=logger, fractions=fractions, threshold=0.4, max_iter=10
        )
        log_data = np.log(fractions)
        v_m = model.compute_variation_matrix(log_data)
        assert v_m.shape == (3, 3)
        assert np.allclose(v_m, v_m.T, atol=1e-6)  # symmetric matrix

    def test_estimate_component_variations(self,setup_teardown_fractions, logger):
        fractions = setup_teardown_fractions
        T = pd.DataFrame(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
        model = SparccKit(
            logger=logger, fractions=fractions, threshold=0.4, max_iter=10
        )
        t = model.estimate_component_variations(T, exclude_pairs={(0, 2), (2, 0)})
        expected = np.array([1, 4, 3]).reshape(-1, 1)
        assert_allclose(t, expected)

    def test_estimate_basis_variances(self, setup_teardown_fractions, logger):
        fractions = setup_teardown_fractions
        comp_vars = np.array([2.0, 2.5, 3.0])
        model = SparccKit(
            logger=logger, fractions=fractions, threshold=0.4, max_iter=10
        )
        omega2 = model.estimate_basis_variances(component_vars=comp_vars)

        assert omega2.shape == (len(comp_vars),)
        assert np.all(omega2 >= 0)

    def test_compute_basis_correlations(self, setup_teardown_fractions, logger):
        fractions = setup_teardown_fractions
        T = np.array([[0, 1, 2],
                      [1, 0, 1],
                      [2, 1, 0]])
        omega2 = np.array([1.0, 2.0, 3.0])
        model = SparccKit(
            logger=logger, fractions=fractions, threshold=0.4, max_iter=10
        )
        b_c = model.compute_basis_correlations(T, omega2)
        assert b_c.shape == (3, 3)
        assert np.allclose(b_c, b_c.T, atol=1e-6)
        assert np.all(np.diag(b_c) == 1)

    def test_run(self, logger, tmp_path, use_tmp_path: bool = True):

        estimated_correlation_matrix_file = "test_sparcckit_correlation_matrix_output.csv"
        if use_tmp_path:
            estimated_correlation_matrix_file = tmp_path / estimated_correlation_matrix_file
            corr_file_name_path, cov_file_name_path, data_file_name_path = set_up_compositional_data_paths(
                result_dir=tmp_path
            )
            simulate_microbiome_data(result_dir=tmp_path)
        else:
            corr_file_name_path, cov_file_name_path, data_file_name_path = set_up_compositional_data_paths()
            simulate_microbiome_data()

        fractions = pd.read_csv(data_file_name_path, sep='\t', index_col=0)
        model = SparccKit(
            logger=logger, fractions=fractions, threshold=0.2, max_iter=100
        )
        model.run()
        corr = model.get_corr_file()
        corr.to_csv(estimated_correlation_matrix_file)

        # Load matrices
        true_corr = pd.read_csv(corr_file_name_path, sep='\t', index_col=0)
        estimated_corr = pd.read_csv(estimated_correlation_matrix_file, index_col=0)

        # Cast to numpy arrays for calculations
        true_corr = true_corr.to_numpy()
        estimated_corr = estimated_corr.to_numpy()

        # Extract upper triangles (excluding diagonal)
        triu_idx = np.triu_indices_from(true_corr, k=1)
        true_vals = true_corr[triu_idx]
        est_vals = estimated_corr[triu_idx]

        mae = np.mean(np.abs(true_vals - est_vals))
        rmse = np.sqrt(np.mean((true_vals - est_vals) ** 2))
        true_estimated_corr = np.corrcoef(true_vals, est_vals)[0, 1]

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Correlation (upper triangle): {true_estimated_corr:.4f}")

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(true_vals, est_vals, alpha=0.1)
        plt.plot([-1, 1], [-1, 1], "r--")
        plt.xlabel("True correlation")
        plt.ylabel("Estimated correlation")
        plt.title("Correlation Matrix Comparison")
        plt.text(0.05, 0.95, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nCorr: {true_estimated_corr:.4f}\nIter: {model.max_iter}",
                 transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

        # Store the performance metrics
        test_results = "test_sparcckit_correlation_matrix_metrics.csv"
        if not use_tmp_path:
            if not os.path.exists(test_results):
                with open(test_results, 'w') as f:
                    f.write("MAE,RMSE,Corr,Iter\n")
            with open(test_results, 'a') as f:
                f.write(f"{mae:.4f},{rmse:.4f},{true_estimated_corr:.4f},{model.max_iter}\n")

        plt.grid(True)
        plt.show()

        assert estimated_corr.shape == (fractions.shape[1], fractions.shape[1])
        assert np.all(estimated_corr <= 1.0)
        assert np.all(estimated_corr >= -1.0)
        assert np.allclose(np.diag(estimated_corr), 1.0)
        assert true_estimated_corr > 0.75
