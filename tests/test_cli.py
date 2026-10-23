import os
import shutil
import subprocess
import sys
from unittest import mock

import pytest

from src.compores import compores_main
from src.compores.__main__ import main
from src.compores.utils import load_configuration, resolve_configuration_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FULL_CONFIG = os.path.join(REPO_ROOT, "configs", "config_synthetic_data.yaml")
QUICK_CONFIG = os.path.join(REPO_ROOT, "configs", "config_synthetic_data_quick.yaml")
COMPORES_COMMAND = shutil.which("compores", path=os.path.dirname(sys.executable))


@pytest.fixture
def runner_class(monkeypatch, tmp_path):
    """Replaces the analysis with a mock, and runs the test in an empty working directory."""
    monkeypatch.chdir(tmp_path)
    runner_class = mock.MagicMock()
    monkeypatch.setattr(compores_main, "ComporesMain", runner_class)
    return runner_class


def dispatch(runner_class, argv):
    """Runs the command with the analysis mocked; returns the selected config file, the runner options and the calls
    made on the runner."""
    main(argv)
    runner_class.assert_called_once()
    config_file_path = os.path.realpath(runner_class.call_args.args[0])
    return config_file_path, runner_class.call_args.kwargs, list(runner_class.return_value.mock_calls)


class TestHelpAndErrors:
    @pytest.mark.parametrize("argv", [
        [], ["-h"], ["--help"], ["--demo", "--help"], ["--help", "--demo", "--config", FULL_CONFIG]
    ])
    def test_help_exits_without_analysis(self, runner_class, tmp_path, capsys, argv):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 0
        stdout = capsys.readouterr().out
        assert "usage:" in stdout and "--demo" in stdout and "--config" in stdout
        runner_class.assert_not_called()
        assert os.listdir(tmp_path) == []

    def test_no_arguments_from_the_command_line(self, runner_class, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["compores"])
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 0
        assert "usage:" in capsys.readouterr().out
        runner_class.assert_not_called()

    def test_empty_argument_list_ignores_the_command_line(self, runner_class, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["compores", "--demo"])
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 0
        runner_class.assert_not_called()

    @pytest.mark.parametrize("argv, message", [
        (["--demo", "--config", FULL_CONFIG], "not allowed with argument"),
        (["--config", FULL_CONFIG, "--demo"], "not allowed with argument"),
        (["--config"], "expected one argument"),
        (["--unknown_flag"], "unrecognized arguments"),
        (["--demo", "extra_argument"], "unrecognized arguments"),
        (["--demo", "--deduplicate", "maybe"], "expected True or False"),
        (["--demo", "--ocu_case", "five"], "invalid int value"),
    ])
    def test_argument_errors_exit_without_analysis(self, runner_class, tmp_path, capsys, argv, message):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
        runner_class.assert_not_called()
        assert os.listdir(tmp_path) == []

    def test_missing_config_file_does_not_fall_back_to_the_demo(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            main(["--config", str(tmp_path / "missing_config.yaml")])
        assert os.listdir(tmp_path) == []

    def test_demo_without_bundled_configuration(self, runner_class, monkeypatch, tmp_path, capsys):
        # As in an installation without the repository folder; the help must still work
        monkeypatch.setattr(compores_main, "CONFIG_FILE_PATH", str(tmp_path / "configs" / "missing.yaml"))
        with pytest.raises(SystemExit) as exit_info:
            main(["--demo"])
        assert exit_info.value.code == 1
        assert "pip install -e ." in capsys.readouterr().err
        runner_class.assert_not_called()
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        assert exit_info.value.code == 0


class TestConfigSelection:
    def test_demo_runs_the_full_synthetic_configuration(self, runner_class):
        config_file_path, options, calls = dispatch(runner_class, ["--demo"])
        assert config_file_path == os.path.realpath(FULL_CONFIG)
        assert options == {"ocu_case": None, "deduplicate": False}
        assert calls == [mock.call.run(only_preprocess_step_flag=False), mock.call.close()]

    def test_demo_is_the_explicit_full_configuration(self, runner_class):
        demo_dispatch = dispatch(runner_class, ["--demo"])
        runner_class.reset_mock()
        assert dispatch(runner_class, ["--config", FULL_CONFIG]) == demo_dispatch

    def test_demo_configuration_resolves_from_any_folder(self, runner_class):
        # The working directory is an empty temporary folder, outside the repository
        config_file_path, _, _ = dispatch(runner_class, ["--demo"])
        config = load_configuration(config_file_path)
        assert resolve_configuration_paths(config, config_file_path) == []
        data_path = os.path.join(REPO_ROOT, "tests", "data", "artificial_experiment_data")
        assert config["PATH_TO_MICROBIOME"] == os.path.join(data_path, "microbiome")
        assert config["PATH_TO_RESPONSE"] == os.path.join(data_path, "response")
        assert config["PATH_TO_OUTPUTS"] == os.path.join(REPO_ROOT, "artificial_experiment_output")
        assert (config["OCU_SAMPLING_RATE"], config["N_SHUFFLES"], config["SHUFFLE_CYCLES"]) == (1, 10, 5)

    def test_quick_configuration_is_kept(self, runner_class):
        config_file_path, _, _ = dispatch(runner_class, ["--config", QUICK_CONFIG])
        assert config_file_path == os.path.realpath(QUICK_CONFIG)

    def test_options_without_config_or_demo_run_the_demo(self, runner_class, capsys):
        config_file_path, _, calls = dispatch(runner_class, ["--preprocess"])
        assert config_file_path == os.path.realpath(FULL_CONFIG)
        assert calls == [mock.call.run(only_preprocess_step_flag=True), mock.call.close()]
        assert "--demo" in capsys.readouterr().err


RUN_OPTIONS = {
    "none": ([], {}, [mock.call.run(only_preprocess_step_flag=False)]),
    "preprocess": (["--preprocess"], {}, [mock.call.run(only_preprocess_step_flag=True)]),
    "preprocess_ignores_tracing": (
        ["--preprocess", "--include_otu_trace_analysis"], {}, [mock.call.run(only_preprocess_step_flag=True)]
    ),
    "tracing_all_responses": (
        ["--include_otu_trace_analysis"], {},
        [mock.call.run(only_preprocess_step_flag=False), mock.call.generate_otu_p_value_summary_data(None)]
    ),
    "tracing_one_response": (
        ["--include_otu_trace_analysis", "--response", "response_5"], {},
        [mock.call.run(only_preprocess_step_flag=False), mock.call.generate_otu_p_value_summary_data("response_5")]
    ),
    "tracing_off": (["--include_otu_trace_analysis", "False"], {}, [mock.call.run(only_preprocess_step_flag=False)]),
    "classification_top_response": (
        ["--include_classification_power_analysis"], {},
        [mock.call.run(only_preprocess_step_flag=False), mock.call.add_synthetic_data_analysis(None)]
    ),
    "classification_one_response": (
        ["--include_classification_power_analysis", "True", "--response", "response_1"], {},
        [mock.call.run(only_preprocess_step_flag=False), mock.call.add_synthetic_data_analysis("response_1")]
    ),
    "tracing_and_classification": (
        ["--include_otu_trace_analysis", "--include_classification_power_analysis", "--response", "response_2"], {},
        [
            mock.call.run(only_preprocess_step_flag=False),
            mock.call.generate_otu_p_value_summary_data("response_2"),
            mock.call.add_synthetic_data_analysis("response_2")
        ]
    ),
    "response_alone": (["--response", "response_1"], {}, [mock.call.run(only_preprocess_step_flag=False)]),
    "deduplicate": (["--deduplicate"], {"deduplicate": True}, [mock.call.run(only_preprocess_step_flag=False)]),
    "deduplicate_off": (
        ["--deduplicate", "False"], {"deduplicate": False}, [mock.call.run(only_preprocess_step_flag=False)]
    ),
    "no_plotting": (
        ["--no_plotting"], {}, [mock.call.switch_off_plotting(), mock.call.run(only_preprocess_step_flag=False)]
    ),
    "ocu_case": (["--ocu_case", "5"], {"ocu_case": 5}, [mock.call.run(only_preprocess_step_flag=False)]),
}
SELECTIONS = {"demo": (["--demo"], FULL_CONFIG), "config": (["--config", QUICK_CONFIG], QUICK_CONFIG),
              "legacy_default": ([], FULL_CONFIG)}


@pytest.mark.parametrize("selection_name, run_option_name", [
    (selection_name, run_option_name) for selection_name in SELECTIONS for run_option_name in RUN_OPTIONS
    # Without any argument, the help is shown (tested above)
    if (selection_name, run_option_name) != ("legacy_default", "none")
])
def test_run_options_are_forwarded(runner_class, selection_name, run_option_name):
    selection_args, expected_config = SELECTIONS[selection_name]
    run_args, expected_options, expected_calls = RUN_OPTIONS[run_option_name]
    config_file_path, options, calls = dispatch(runner_class, selection_args + run_args)
    assert config_file_path == os.path.realpath(expected_config)
    assert options == {"ocu_case": None, "deduplicate": False, **expected_options}
    assert calls == expected_calls + [mock.call.close()]


ENTRY_POINTS = [
    pytest.param([sys.executable, "-m", "compores"], id="python -m compores"),
    pytest.param([COMPORES_COMMAND], id="compores", marks=pytest.mark.skipif(
        COMPORES_COMMAND is None, reason="the `compores` command is not installed; run `pip install -e .`"
    )),
]


@pytest.mark.parametrize("command", ENTRY_POINTS)
@pytest.mark.parametrize("argv, exit_code, expected_output", [
    ([], 0, "usage:"),
    (["--help"], 0, "--demo"),
    (["--demo", "--config", FULL_CONFIG], 2, "not allowed with argument"),
    (["--unknown_flag"], 2, "unrecognized arguments"),
])
def test_entry_points(tmp_path, command, argv, exit_code, expected_output):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH")]))
    result = subprocess.run(command + argv, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == exit_code
    assert expected_output in (result.stdout if exit_code == 0 else result.stderr)
    assert os.listdir(tmp_path) == []


def test_help_does_not_load_the_analysis(tmp_path):
    code = (
        "import sys\n"
        "from compores.__main__ import main\n"
        "try:\n"
        "    main([])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('compores.compores_main' in sys.modules, 'numpy' in sys.modules)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(REPO_ROOT, "src"), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip().splitlines()[-1] == "False False"
