import os

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.post_processing.plotting.pyplot_plotting import PyPlotPlotter
from vv.evaluation.post_processing.plotting.result_loader import EvaluationResultsLoader
from vv.evaluation.post_processing.plotting.tensorBoard import TensorBoardPlotter


class EvaluationVisualizer:
    def __init__(self,
                 destination_path: str) -> None:
        """
        Initialize the EvaluationPlotter.

        Parameters
        ----------
        destination_path : str
            Path to the folder containing the evaluation results.
        """
        self.destination_path_eval = os.path.join(os.path.abspath(destination_path), eval_def.EVAL_FOLDER_NAME)
        self.results_path = os.path.join(self.destination_path_eval, eval_def.PRCESSED_RESULTS_FOLDER_NAME)
        if not os.path.exists(self.destination_path_eval):
            raise ValueError(f"Destination path {self.destination_path_eval} does not exist.")
        os.makedirs(self.results_path, exist_ok=True)

        self.data: dict = {}
        self.result_loader = EvaluationResultsLoader(self.destination_path_eval)
        self.pyplot_plotter = PyPlotPlotter(self.results_path)
        self.tensorBoard_plotter = TensorBoardPlotter(self.results_path)

    def load_results(self) -> None:
        """
        Load results from the destination path.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.data = self.result_loader.load_results()

    def plot_metrics(self) -> None:
        """
        Plot metrics from the loaded results.
        Save plots in top level dir

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_results() first.")

        self.pyplot_plotter.plot_metrics(self.data)

    def plot_latest_codebook_usage(self) -> None:
        """
        Plot the latest codebook usage from the loaded results.
        Save histograms in the top-level directory.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_results() first.")

        self.pyplot_plotter.plot_latest_codebook_usage(self.data)

    def save_codebook_statistics_as_csv(self) -> None:
        """
        Save codebook statistics as CSV files.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_results() first.")

        self.pyplot_plotter.save_codebook_statistics_as_csv(self.data)

    def log_to_tensorboard(self) -> None:
        """
        Log evaluation results to TensorBoard.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_results() first.")

        self.tensorBoard_plotter.log_to_tensorboard(self.data)
