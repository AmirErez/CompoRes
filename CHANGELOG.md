# Changelog

## 0.3.0 (2026-10-23)

Changes since 0.2.0.

### Changed

- `compores` without arguments now shows the help and exits, instead of starting the full synthetic demo. Run the
  demo with the new `--demo` option, `compores --demo`, from any folder; from the `CompoRes` folder,
  `compores --config configs/config_synthetic_data.yaml` is the same run. Like every run, `--demo` resumes from the
  results already in `artificial_experiment_output` instead of starting over; delete that folder to rerun the demo
  from scratch.
- `--demo` and `--config` cannot be combined, and `--config` needs a path.
- For backward compatibility, options given without `--config` or `--demo`, e.g.
  `compores --include_otu_trace_analysis`, still run the demo, now with a notice. Give `--demo` or `--config`
  explicitly in new commands and scripts.
- Relative paths in a configuration file are now read relative to the folder of the configuration file, so CompoRes
  runs from any folder. Configuration files written for earlier versions, with paths relative to the folder they were
  run from, still work from that folder, with a warning in the log.
- Installation: a single environment file, `env/compores_env.yml`, pins the tested dependency versions (Python 3.12)
  for Linux and macOS, Apple Silicon included; `mkl`, which is not available for Apple Silicon, is no longer
  installed. `pip install -e .` installs CompoRes into the environment and adds the `compores` command.
- Outlier removal is off unless `OUTLIERS_REMOVAL` is set in the configuration file. It is experimental, and no longer
  documented or set in the bundled configuration files.

### Added

- The `compores` command, and the `--demo` option.
- A heatmap of the SparCC taxon correlations, ordered by the hierarchical clustering that the OCUs are cut from,
  with its dendrogram, in `plots/taxa_correlation_clustering/`.
- `report.html`, written to the output folder at the end of every run: an overview of the results to open in a
  browser, with the response ranking, the plots and best balances of every response, the taxon clustering heatmap,
  and links to all result files.
- A quick synthetic test, `configs/config_synthetic_data_quick.yaml`, about 10 times faster than the full demo.
- Documentation: a restructured README, a synthetic-data tutorial (`docs/tutorial.md`) and a reference
  (`docs/reference.md`).

### Fixed

- Runs failed with pandas 3 (read-only arrays under Copy-on-Write; `DataFrame.stack` keeping missing values), and the
  power analysis failed with NumPy 2.4 and later, which removed `np.trapz`; it also no longer uses the `vert` option
  of matplotlib's box plots, which matplotlib 3.13 removes.
- `--deduplicate`, `--include_otu_trace_analysis` and `--include_classification_power_analysis` treated `False` as
  true; they now take the flag alone, or `True`/`False`.
- The power analysis failed on macOS, whose bash 3.2 lacks `wait -n`. When the power-analysis script fails, CompoRes
  now stops with an error instead of reporting it as finished.

### Tests and CI

- `pytest` runs the fast tests by default, including an end-to-end run of the quick synthetic test and tests of the
  command line; the slow power-analysis test is optional (`pytest -m slow`, and in CI the manually started Slow Tests
  workflow).
- CI runs on Python 3.12 with the pinned environment. It runs when the code, tests, configuration files, environment
  file or packaging change, selected by GitHub's `paths` filter instead of the third-party `tj-actions/changed-files`
  action, and now fails when tests fail (the exit status of `pytest` was hidden by `tee`).
- Running the tests from the repository folder no longer deletes the demo results in `artificial_experiment_output`,
  and a second test run no longer fails on the test outputs left by the first.
- The tests no longer open plot windows on macOS, which could make the test run wait for them to be closed.
