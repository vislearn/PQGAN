import argparse

import vv.execution as execution
from vv.evaluation import EvalExecutionManager, EvaluationVisualizer
from vv.utilities import str2bool


def main(argv: str = None) -> None:
    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Vector quantization for image autoencoders.')
    subparsers = parser.add_subparsers(dest='command_name')

    eval_latent_space = subparsers.add_parser("show-latent", help="Visualize the latent space of a model")  # noqa: F841
    fit_parser = subparsers.add_parser("fit")  # noqa: F841
    val_parser = subparsers.add_parser("validate")  # noqa: F841
    test_parser = subparsers.add_parser("test")  # noqa: F841

    evaluate = subparsers.add_parser("eval", help="Visualize the latent space of a model")
    evaluate.add_argument("--source_path", type=str, help="Path to the model checkpoint")
    evaluate.add_argument("--destination_path", type=str, help="Path to where the evaluation results will be saved")
    evaluate.add_argument("--eval_config", type=str, help="Path to the evaluation configuration file")
    evaluate.add_argument("--overwrite_possible", type=str2bool, help="Overwrite existing results", default=False)

    plot = subparsers.add_parser("plot", help="Visualize the latent space of a model")
    plot.add_argument("--destination_path", type=str, help="Path to where the evaluation results are saved")

    # Evaluate input
    args, unknown = parser.parse_known_args(argv)

    match args.command_name:
        case "eval":
            folderManager = EvalExecutionManager(args.source_path,
                                                 args.destination_path,
                                                 args.eval_config,
                                                 args.overwrite_possible)
            folderManager.run_evaluation()
        case "plot":
            # Plotting
            plotter = EvaluationVisualizer(args.destination_path)
            plotter.load_results()
            plotter.plot_metrics()
            plotter.save_codebook_statistics_as_csv()
            plotter.plot_latest_codebook_usage()
            plotter.log_to_tensorboard()

        case "show-latent":
            print("Hello World")
        case _:
            # Lightning CLI
            execution.cli_main(argv)
            # Custom debug main
            # execution.debug_main(argv)
