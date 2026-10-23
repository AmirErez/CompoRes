import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.compores import compores_plotting
from src.compores.compores_plotting import delete_older_shuffle_files, plot_taxa_correlation_clustered_heatmap, \
    taxa_correlation_heatmap_path


class TestComporesPlotting:
    # Files with fewer shuffles than current_shuffles are deleted
    def test_delete_older_shuffle_files(self, tmp_path):
        # Create a temporary directory
        tmp_dir = tmp_path / "shuffle_files"
        tmp_dir.mkdir()

        # Create shuffle files with fewer shuffles than current_shuffles
        shuffle_files = [
            "pl_5_shuffles.tsv",
            "plot_7_shuffles.png",
            "plot_10_shuffles.png",
            "plot_15.png", # files with no shuffles word in the name are not considered
        ]

        for shuffle_file in shuffle_files:
            with open(tmp_dir / shuffle_file, "w") as f:
                f.write("")

        # Call delete_older_shuffle_files
        current_shuffles = 8
        delete_older_shuffle_files(tmp_dir, current_shuffles)

        # Check that shuffle files with fewer shuffles are deleted and the rest are kept
        assert len(os.listdir(tmp_dir)) == 2
        assert "pl_5_shuffles.tsv" not in os.listdir(tmp_dir)
        assert "plot_7_shuffles.png" not in os.listdir(tmp_dir)
        assert "plot_10_shuffles.png" in os.listdir(tmp_dir)
        assert "plot_15.png" in os.listdir(tmp_dir)

    @staticmethod
    def correlations_and_linkage(n_taxa):
        values = np.corrcoef(np.random.default_rng(0).normal(size=(30, n_taxa)), rowvar=False)
        values = (values + values.T) / 2
        taxa = [f"OTU_{i}" for i in range(n_taxa)]
        correlations = pd.DataFrame(values, index=taxa, columns=taxa)
        distances = np.clip(0.5 * (1 - values), 0, None)
        np.fill_diagonal(distances, 0)
        return correlations, linkage(squareform(distances), method="average")

    def test_plot_taxa_correlation_clustered_heatmap(self, tmp_path):
        path_to_plot = taxa_correlation_heatmap_path(tmp_path, "a-b-c")
        assert plot_taxa_correlation_clustered_heatmap(*self.correlations_and_linkage(8), "a-b-c", path_to_plot)
        assert os.path.getsize(path_to_plot) > 0

    def test_no_heatmap_of_too_many_taxa(self, tmp_path, monkeypatch):
        monkeypatch.setattr(compores_plotting, "MAX_HEATMAP_TAXA", 5)
        path_to_plot = taxa_correlation_heatmap_path(tmp_path, "a-b-c")
        assert not plot_taxa_correlation_clustered_heatmap(*self.correlations_and_linkage(8), "a-b-c", path_to_plot)
        assert not os.path.exists(path_to_plot)
