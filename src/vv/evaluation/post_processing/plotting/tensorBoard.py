import glob
import os
from typing import Dict

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.core.util import codebook_statistics_summary
from vv.shared.shared_types import CodebookUsageEntry


class TensorBoardPlotter:
    tensorboard_log_folder_name = "tensorboard_log"

    def __init__(self,
                 results_path: str) -> None:
        """
        Initialize the TensorBoardPlotter.

        Parameters
        ----------
        results_path : str
            Path to the directory where results are stored.
        """
        self.results_path = results_path
        self.tensorboard_log_dir = os.path.join(self.results_path,
                                                TensorBoardPlotter.tensorboard_log_folder_name)

    def log_to_tensorboard(self,
                           data: Dict[str, pd.DataFrame]) -> None:
        """
        Log evaluation results to TensorBoard.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing the evaluation results.
            The keys are model names and the values are DataFrames with the data.

        Returns
        ----------
        None
        """
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

        pbar = tqdm(data.items(), unit="model")
        for model_name, df in pbar:
            pbar.set_description(f"Logging {model_name} to TensorBoard")
            self._log_model_to_tensorboard(model_name, df)

        print(f"TensorBoard logs written to: {self.tensorboard_log_dir}")

    def _log_model_to_tensorboard(self,
                                  model_name: str,
                                  df: pd.DataFrame) -> None:
        """
        Log metrics and other information for a specific model to TensorBoard.

        Parameters
        ----------
        model_name : str
            Name of the model.
        df : pd.DataFrame
            DataFrame containing the metrics and other information.

        Returns
        ----------
        None
        """
        model_log_dir = os.path.join(self.tensorboard_log_dir, model_name)
        self._clear_old_tensorboard_logs(model_log_dir)
        os.makedirs(model_log_dir, exist_ok=True)

        df = df.sort_values(by=eval_def.STEP_KEY_NAME)
        with SummaryWriter(log_dir=model_log_dir) as writer:
            self._write_metrics_to_tensorboard(model_name,
                                               writer,
                                               df)

    def _clear_old_tensorboard_logs(self,
                                    model_log_dir: str) -> None:
        """
        Remove existing TensorBoard log files for a model.

        Parameters
        ----------
        model_log_dir : str
            Path to the model's TensorBoard log directory.

        Returns
        ----------
        None
        """
        for f in glob.glob(os.path.join(model_log_dir, "events.out.tfevents.*")):
            os.remove(f)

    def _write_metrics_to_tensorboard(self,
                                      model_name: str,
                                      writer: SummaryWriter,
                                      df: pd.DataFrame) -> None:
        """
        Write metrics from DataFrame to TensorBoard.

        Parameters
        ----------
        model_name : str
            Name of the model.
        writer : SummaryWriter
            TensorBoard writer object.
        df : pd.DataFrame
            DataFrame containing the metrics.

        Returns
        ----------
        None
        """
        self._write_scalar_metrics_to_tensorboard(model_name, writer, df)
        self._write_codebook_usage_to_tensorboard(model_name, writer, df)

    def _write_scalar_metrics_to_tensorboard(self,
                                             model_name: str,
                                             writer: SummaryWriter,
                                             df: pd.DataFrame) -> None:
        """
        Write scalar metrics from DataFrame to TensorBoard.

        Parameters
        ----------
        model_name : str
            Name of the model.
        writer : SummaryWriter
            TensorBoard writer object.
        df : pd.DataFrame
            DataFrame containing the metrics.

        Returns
        ----------
        None
        """
        skip_columns = {
            eval_def.STEP_KEY_NAME,
            eval_def.TIMESTAMP_KEY_NAME,
            # Non float columns
            *[col for col in df.columns if not pd.api.types.is_float_dtype(df[col])]
        }

        for column in df.columns:
            if column in skip_columns:
                continue

            for step, value in zip(df[eval_def.STEP_KEY_NAME], df[column]):
                try:
                    if pd.isna(value) or pd.isna(step):
                        continue
                    writer.add_scalar(column, value, int(step))
                except Exception as e:
                    tqdm.write(f"Warning: Error adding scalar metrics for {model_name} at step {step}: {e}")

    def _write_codebook_usage_to_tensorboard(self,
                                             model_name: str,
                                             writer: SummaryWriter,
                                             df: pd.DataFrame) -> None:
        """
        Write codebook usage metrics from DataFrame to TensorBoard.

        Parameters
        ----------
        model_name : str
            Name of the model.
        writer : SummaryWriter
            TensorBoard writer object.
        df : pd.DataFrame
            DataFrame containing the metrics.

        Returns
        ----------
        None
        """
        if eval_def.CODEBOOK_STATISTICS_KEY_NAME not in df.columns:
            return

        for step, codebook_statistics in zip(df[eval_def.STEP_KEY_NAME], df[eval_def.CODEBOOK_STATISTICS_KEY_NAME]):
            if pd.isna(step):  # Check if the step is NaN
                continue

            try:
                # Check if codebook_usage is a list of tensors and if any tensor contains NaN
                if not isinstance(codebook_statistics, list) or not all(isinstance(entry, CodebookUsageEntry) for entry in codebook_statistics):
                    if codebook_statistics is None or pd.isna(codebook_statistics):
                        raise ValueError("No codebook statistics found.")
                    else:
                        raise ValueError(f"Expected CodebookUsage, got {type(codebook_statistics)}")

                if len(codebook_statistics) == 0:
                    raise ValueError("Codebook statistics list is empty.")

                if any(entry.bincount.cpu().isnan().any() for entry in codebook_statistics):
                    raise ValueError("Codebook statistics contain NaN values.")

                codebook_metrics = codebook_statistics_summary(codebook_statistics)
                codebook_binary_usage = np.array(codebook_metrics["utilization_per_subcodebook"])
                avg_util_rate = codebook_metrics["avg_utilization"]
                normalized_entropy = codebook_metrics["normalized_entropy"]
                normalized_perplexity = codebook_metrics["normalized_perplexity"]

                # Define the number of bins for the histogram
                num_bins = 20  # You can adjust this value based on your data
                hist, bin_edges = np.histogram(codebook_binary_usage, bins=num_bins, range=(0, 1))
                bin_edges = bin_edges[1:]

                writer.add_histogram_raw(
                    tag="Codebook_Utilization",
                    min=0.0,
                    max=1.0,
                    num=len(hist),
                    sum=float(sum(hist)),
                    sum_squares=float(sum(v ** 2 for v in hist)),
                    bucket_limits=bin_edges,
                    bucket_counts=hist,
                    global_step=int(step)
                )

                writer.add_scalar("Codebook_Non_Zero_Usage_Fraction", avg_util_rate, int(step))
                writer.add_scalar("Codebook_Normalized_Entropy", normalized_entropy, int(step))
                writer.add_scalar("Codebook_Normalized_Perplexity", normalized_perplexity, int(step))
            except Exception as e:
                tqdm.write(f"Warning: Error adding codebook statistics for model {model_name} at step {step}: {e}")
                continue
