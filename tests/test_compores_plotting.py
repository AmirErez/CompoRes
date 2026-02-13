import os

from src.compores.compores_plotting import delete_older_shuffle_files


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
