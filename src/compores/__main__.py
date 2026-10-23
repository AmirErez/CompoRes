import argparse
import os
import sys

# Wrapped by hand, to fit an 80-column terminal
DESCRIPTION = """\
`CompoRes` is an algorithmic package designed to analyze correlation between
changes in the host's microbiota and the host's gene expression. It addresses
two significant analysis challenges: compositional statistics in the microbiota
and multiple hypothesis testing, which can be a source of spurious correlations
between microbiota and host."""

EPILOG = """\
examples:
  compores --config my_config.yaml   run CompoRes on your data
  compores --demo                    run the full synthetic demo
  compores --config configs/config_synthetic_data_quick.yaml
                                     run the quick synthetic test (from the
                                     CompoRes folder)

Without arguments, compores shows this help. For backward compatibility, other
options given without --config or --demo run the demo, as in earlier versions."""


def str_to_bool(value: str) -> bool:
    """Parses a command-line boolean; `type=bool` would turn any non-empty string, "False" included, into True."""
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    input_selection = parser.add_mutually_exclusive_group()
    input_selection.add_argument(
        '--config',
        type=str,
        default=None,
        help='The path to the config file of your analysis.'
    )
    input_selection.add_argument(
        '--demo',
        action='store_true',
        help='Run the full synthetic demo, configured by `configs/config_synthetic_data.yaml`; the results go to '
             '`artificial_experiment_output` in the CompoRes folder, and a rerun resumes from them.'
    )
    parser.add_argument(
        '--include_otu_trace_analysis',
        nargs='?',
        type=str_to_bool,
        const=True,
        default=False,
        help='Add OTU p-value tracing and analysis after the run (the flag alone, or True/False).'
    )
    parser.add_argument(
        '--include_classification_power_analysis',
        nargs='?',
        type=str_to_bool,
        const=True,
        default=False,
        help='Add derived synthetic response generation and classification after the run (the flag alone, or '
             'True/False).'
    )
    parser.add_argument(
        '--response',
        nargs='?',
        type=str,
        default=None,
        help='The response variable to OTU p-value tracing or to apply synthetic power analysis to.'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run only the preprocessing step and exit.'
    )
    parser.add_argument(
        '--no_plotting',
        action='store_true',
        help='Do not plot the results, nor write the results report, `report.html`.'
    )
    parser.add_argument(
        '--ocu_case',
        nargs='?',
        type=int,
        default=None,
        help='The OCU case to analyze; parameter in Synthetic Power Analysis.'
    )
    parser.add_argument(
        '--deduplicate',
        nargs='?',
        type=str_to_bool,
        const=True,
        default=False,
        help='Deduplicate the input automatically, keeping the first entry of duplicated sample tags (the flag alone, '
             'or True/False).'
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Runs the `compores` command with the given arguments, by default those of the command line."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    # Without arguments, show the help rather than start a long analysis
    args = parser.parse_args(argv or ['--help'])

    # Imported only here, so that the help and the argument errors do not wait for the analysis modules to load
    from .compores_main import ComporesMain, CONFIG_FILE_PATH

    config_file_path = args.config
    if config_file_path is None:
        if not os.path.isfile(CONFIG_FILE_PATH):
            parser.exit(1, f"{parser.prog}: error: the demo configuration {os.path.normpath(CONFIG_FILE_PATH)} was not "
                           f"found; the demo needs CompoRes installed from its repository folder with "
                           f"`pip install -e .`\n")
        if not args.demo:
            print(f"{parser.prog}: neither --config nor --demo was given, so the demo runs, as in earlier versions; "
                  f"give --demo or --config explicitly", file=sys.stderr)
        config_file_path = CONFIG_FILE_PATH

    runner = ComporesMain(config_file_path, ocu_case=args.ocu_case, deduplicate=args.deduplicate)
    if args.no_plotting:
        runner.switch_off_plotting()
    try:
        runner.run(only_preprocess_step_flag=args.preprocess)
        if not args.preprocess:
            if args.include_otu_trace_analysis:
                runner.generate_otu_p_value_summary_data(args.response)
            if args.include_classification_power_analysis:
                runner.add_synthetic_data_analysis(args.response)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
