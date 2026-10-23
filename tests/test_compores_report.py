import os
import re
from collections import Counter
from unittest import mock

import pandas as pd
import pytest

from src.compores import compores_main, compores_report
from src.compores.compores_main import ComporesMain
from src.compores.compores_report import format_p_value, format_taxa, write_results_report, RANKING


@pytest.mark.parametrize("p_value, expected", [(float("nan"), "–"), (1e-11, "≤ 1e-11"), (0.0909, "0.091")])
def test_format_p_value(p_value, expected):
    assert format_p_value(p_value) == expected


def test_format_taxa():
    assert format_taxa("['OTU_5', 'OTU_24']") == ("OTU_5, OTU_24", "OTU_5, OTU_24")
    short_text, full_text = format_taxa(str([f"OTU_{i}" for i in range(1, 9)]))
    assert short_text.startswith("8 taxa: OTU_1, ") and short_text.endswith(", …")
    assert full_text.endswith("OTU_8")
    assert format_taxa(float("nan")) == ("–", "")


def write_ranking(outputs_path, experiment, method, responses):
    folder = outputs_path / RANKING / experiment / method
    folder.mkdir(parents=True)
    pd.DataFrame(
        {"gev": range(len(responses), 0, -1), "bootstrap": [1.0] * len(responses)}, index=responses
    ).to_csv(folder / "minus_mean_log_p_value_sorted_responses_at_shuffling_cycle_1.csv")


def test_no_report_without_ranking(tmp_path):
    assert write_results_report(str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_report_of_several_experiments(tmp_path):
    for experiment in ("a-b-c", "a-b-d"):
        write_ranking(tmp_path, experiment, "CLR", ["response_1_x", "response_2_y"])
    page = open(write_results_report(str(tmp_path)), encoding="utf-8").read()
    assert "a-b-c" in page and "a-b-d" in page
    duplicated_ids = [i for i, count in Counter(re.findall(r'\sid="([^"]+)"', page)).items() if count > 1]
    assert duplicated_ids == []


def test_links_to_names_with_special_characters(tmp_path):
    experiment, response = "a-b-c", "response_1_IL 6#2%"
    write_ranking(tmp_path, experiment, "CLR", [response])
    plot_folder = tmp_path / "plots" / "compores_signal_significance" / experiment / "CLR"
    plot_folder.mkdir(parents=True)
    (plot_folder / f"{experiment}_{response}_10_shuffles.png").write_bytes(b"")
    page = open(write_results_report(str(tmp_path)), encoding="utf-8").read()
    assert 'src="plots/compores_signal_significance/a-b-c/CLR/a-b-c_response_1_IL%206%232%25_10_shuffles.png"' in page


def test_details_of_the_top_ranked_responses_only(tmp_path, monkeypatch):
    monkeypatch.setattr(compores_report, "MAX_DETAILED_RESPONSES", 2)
    write_ranking(tmp_path, "a-b-c", "CLR", ["response_1_x", "response_2_y", "response_3_z"])
    page = open(write_results_report(str(tmp_path)), encoding="utf-8").read()
    assert page.count('<details class="response"') == 2
    assert "top-ranked of the 3 responses" in page and page.count('<tr data-response=') == 3


class TestComporesMainReport:
    @pytest.fixture
    def runner(self, tmp_path):
        runner = ComporesMain.__new__(ComporesMain)
        runner.plotting_flag = True
        runner.config_dict = {"PATH_TO_OUTPUTS": str(tmp_path)}
        runner.logger = mock.MagicMock()
        return runner

    def test_report_errors_do_not_fail_the_run(self, runner, monkeypatch):
        monkeypatch.setattr(compores_main, "write_results_report", mock.MagicMock(side_effect=OSError("disk full")))
        runner.write_report()
        runner.logger.warning.assert_called_once()

    def test_no_report_without_plotting(self, runner, monkeypatch):
        report_writer = mock.MagicMock()
        monkeypatch.setattr(compores_main, "write_results_report", report_writer)
        runner.plotting_flag = False
        runner.write_report()
        report_writer.assert_not_called()
