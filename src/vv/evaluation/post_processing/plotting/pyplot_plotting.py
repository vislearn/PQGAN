import os
import typing as Typing
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.core.util import codebook_statistics_summary
from vv.shared.shared_types import CodebookUsage


class PyPlotPlotter:
    def __init__(self,
                 results_path: str) -> None:
        self.results_path = results_path

    def plot_metrics(self,
                     data: Dict[str, pd.DataFrame]) -> None:
        """
        Plot metrics from given data
        Save plots in top level dir

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing the data to plot.
            The keys are model names and the values are DataFrames with the data.

        Returns
        ----------
        None
        """
        # Get list of all metrics
        all_metrics = self._extract_metric_names(data)

        # Check for missing columns
        for model_name, df in data.items():
            self._check_for_missing_columns(df)

        # Prepare values for plot
        highest_step = max(df[eval_def.STEP_KEY_NAME].max() for df in data.values())

        # Plot each metric
        for metric in tqdm(all_metrics, desc="Plotting metrics", unit="metric"):
            self._plot_metric(metric, data, highest_step)

        print(f"Saved metric plots from {eval_def.RESULTS_FILE_NAME} files to {self.results_path}")

    def plot_latest_codebook_usage(self,
                                   data: Dict[str, pd.DataFrame]) -> None:
        """
        Plot the latest codebook usage from the loaded results.
        Save histograms in the top-level directory.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing the data to plot.
            The keys are model names and the values are DataFrames with the data.

        Returns
        ----------
        None
        """
        # Process each model's data
        for model_name, df in tqdm(data.items(), desc="Plotting codebook statistics with PyPlot", unit="model"):
            # Check if the codebook usage key exists in the DataFrame
            if eval_def.CODEBOOK_STATISTICS_KEY_NAME not in df.columns:
                tqdm.write(f"Warning: {eval_def.CODEBOOK_STATISTICS_KEY_NAME} not found in {model_name}.")
                continue

            # Get the row with the highest step value
            latest_row = df.loc[df[eval_def.STEP_KEY_NAME].idxmax()]
            codebook_statistics: CodebookUsage = latest_row[eval_def.CODEBOOK_STATISTICS_KEY_NAME]
            step = latest_row[eval_def.STEP_KEY_NAME]

            try:
                codebook_metrics: dict = codebook_statistics_summary(codebook_statistics)

                # Create a histogram
                self._plot_codebook_statistics(codebook_metrics,
                                               model_name,
                                               step)
            except Exception as e:
                tqdm.write(f"Warning: Error processing codebook usage for {model_name} at step {step}: {e}")
                continue

        print(f"Saved codebook usage histograms to {self.results_path}/{eval_def.CODEBOOK_STATISTICS_KEY_NAME}")

    def save_codebook_statistics_as_csv(self,
                                        data: Dict[str, pd.DataFrame]) -> None:
        """
        Save codebook statistics as CSV files.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing the data to plot.
            The keys are model names and the values are DataFrames with the data.
        Returns
        ----------
        None
        """
        # Create CSV file
        dest_path = os.path.join(self.results_path, eval_def.CODEBOOK_STATISTICS_KEY_NAME)
        os.makedirs(dest_path, exist_ok=True)
        csv_file_path = os.path.join(dest_path, f"{eval_def.CODEBOOK_STATISTICS_KEY_NAME}.csv")
        file_untouched = True

        # Process each model's data
        for model_name, df in tqdm(data.items(), desc="Creating codebook_statistics csv", unit="model"):
            # Check if the codebook usage key exists in the DataFrame
            if eval_def.CODEBOOK_STATISTICS_KEY_NAME not in df.columns:
                tqdm.write(f"Warning: {eval_def.CODEBOOK_STATISTICS_KEY_NAME} not found in {model_name}.")
                continue

            # Get the row with the highest step value
            latest_row = df.loc[df[eval_def.STEP_KEY_NAME].idxmax()]
            codebook_statistics: CodebookUsage = latest_row[eval_def.CODEBOOK_STATISTICS_KEY_NAME]
            step = latest_row[eval_def.STEP_KEY_NAME]

            try:
                codebook_metrics: dict = codebook_statistics_summary(codebook_statistics)

                # Filter only float values
                codebook_metrics = {k: v for k, v in codebook_metrics.items() if isinstance(v, float)}

                # Add to csv file
                if file_untouched:
                    header = "model_name" + "," + eval_def.STEP_KEY_NAME + "," + ",".join(codebook_metrics.keys())
                    with open(csv_file_path, "w") as f:
                        f.write(header + os.linesep)
                    file_untouched = False

                row = model_name + "," + str(step) + "," + ",".join(str(codebook_metrics[k]) for k in codebook_metrics.keys())
                with open(csv_file_path, "a") as f:
                    f.write(row + os.linesep)
            except Exception as e:
                tqdm.write(f"Warning: Error saving codebook usage as csv for {model_name} at step {step}: {e}")
                continue

        print(f"Saved codebook usage csv values to {csv_file_path}")

    def _extract_metric_names(self,
                              data: Dict[str, pd.DataFrame]) -> Typing.List[str]:
        """
        Extract the names of the valid metrics from the DataFrame.

        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing the data to plot.
            The keys are model names and the values are DataFrames with the data.

        Returns
        -------
        list
            List of metric names.
        """
        all_metrics = list(next(iter(data.values())).columns)
        if eval_def.STEP_KEY_NAME in all_metrics:
            all_metrics.remove(eval_def.STEP_KEY_NAME)
            all_metrics.remove(eval_def.TIMESTAMP_KEY_NAME)
            # Remove non-float columns
            all_metrics = [metric for metric in all_metrics if pd.api.types.is_float_dtype(data[next(iter(data))][metric])]
        else:
            raise ValueError(f"Expected {eval_def.STEP_KEY_NAME} column in {eval_def.RESULTS_FILE_NAME}.")

        return all_metrics

    def _check_for_missing_columns(self,
                                   df: pd.DataFrame) -> None:
        """
        Check for missing columns in DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to check.

        Returns
        -------
        None
        """
        if df is not None and eval_def.STEP_KEY_NAME not in df.columns:
            raise RuntimeError(f"'{eval_def.STEP_KEY_NAME}' column is missing in DataFrame.")

    def _plot_metric(self,
                     metric: str,
                     data: Dict[str, pd.DataFrame],
                     highest_step: int) -> None:
        """
        Plot a single metric.

        Parameters
        ----------
        metric : str
            The name of the metric to plot.
        data : Dict[str, pd.DataFrame]
            Dictionary containing the data to plot.
            The keys are model names and the values are DataFrames with the data.
        highest_step : int
            The highest step value across all models.

        Returns
        -------
        None
        """
        # Prepare plot
        all_values = [df[metric].values for df in data.values()]
        y_min = min(min(values) for values in all_values)
        y_max = max(max(values) for values in all_values)
        y_margin = 0.2 * (y_max - y_min)
        y_min -= y_margin
        y_max += y_margin

        # Plot
        plt.figure()
        plt.title(f"{metric} over training steps")

        # Data
        for model_name, df in data.items():
            plt.scatter(df[eval_def.STEP_KEY_NAME], df[metric], label=model_name, s=10)

        # Axis
        plt.xlabel(eval_def.STEP_KEY_NAME)
        plt.xlim(0, 1.2 * highest_step)
        plt.ylabel(metric)
        plt.ylim(y_min, y_max)

        # General
        plt.grid(True)

        # Legend
        num_labels = len(data)
        bottom_margin = min(0.035 * num_labels, 0.3)
        plt.subplots_adjust(bottom=bottom_margin)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False, fontsize=6)

        plt.savefig(os.path.join(self.results_path, f"{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_codebook_statistics(self,
                                  codebook_metrics: dict,
                                  model_name: str,
                                  step: int) -> None:
        """
        Plot the codebook usage.

        Parameters
        ----------
        codebook_metrics : dict
            Dictionary containing the codebook metrics.
        model_name : str
            Name of the model.
        step : int
            The step at which the metrics were collected.

        Returns
        -------
        None
        """
        codebook_binary_usage = codebook_metrics["utilization_per_subcodebook"]
        avg_util_rate = codebook_metrics["avg_utilization"]
        normalized_entropy = codebook_metrics["normalized_entropy"]
        normalized_perplexity = codebook_metrics["normalized_perplexity"]

        plt.figure()
        plt.hist(codebook_binary_usage, bins=20, range=(0, 1), edgecolor='black')
        plt.title(f"Codebook Utilization Histogram for {model_name} at Step {step}")
        plt.xlabel("Utilization Rate")
        plt.ylabel("Frequency")
        plt.text(0.5, 0.9, f"Avg Utilization Rate: {avg_util_rate:.2f}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.85, f"Normalized Entropy: {normalized_entropy:.2f}", transform=plt.gca().transAxes)
        plt.text(0.5, 0.8, f"Normalized Perplexity: {normalized_perplexity:.2f}", transform=plt.gca().transAxes)
        plt.grid(True)

        # Save the histogram
        result_path = os.path.join(self.results_path,
                                   eval_def.CODEBOOK_STATISTICS_KEY_NAME)
        os.makedirs(result_path, exist_ok=True)
        plt.savefig(f"{result_path}/{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
