import typing as Typing
from argparse import Namespace
from datetime import datetime

import torch
from lightning.pytorch.loggers import Logger

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.core.result_data import MetricResult
from vv.shared.shared_types import CodebookUsage, CodebookUsageEntry


class InMemoryLogger(Logger):
    """
    A simple in-memory logger for metrics. That has compatible function signature for MetricsLogger
    """
    def __init__(self) -> None:
        self._log_value: Typing.Any = None
        self._histogram: Typing.Any = None
        self._name = "InMemoryLogger"
        self._version = datetime.now().strftime("%Y%m%d%H%M%S")

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def log_hyperparams(self,
                        params: Typing.Union[Typing.Dict[str, Typing.Any], Namespace],
                        *args: Typing.Any,
                        **kwargs: Typing.Any) -> None:
        """
        Funciton required by lightning

        Parameters:
        ----------
        params: dict or Namespace
            A dictionary or Namespace containing the hyperparameters to log.
            Key is hyperparameter name and value is the hyperparameter value.
        *args: Any
            Additional positional arguments (not used).
        **kwargs: Any
            Additional keyword arguments (not used).

        Returns:
        ----------
        None
        """
        if isinstance(params, Namespace):
            self._hyperparams = vars(params)
        else:
            self._hyperparams = dict(params)

    def log_metrics(self,
                    metrics: Typing.Dict[str, Typing.Any],
                    step: int) -> None:
        """
        Logs the metrics to the in-memory logger.

        Parameters:
        ----------
        metrics: dict
            A dictionary containing the metrics to log.
            Key is metric name and value is the metric value.
        step: int
            The step at which the metrics are logged.

        Returns:
        ----------
        None
        """
        metric_results: MetricResult = []
        for key, value in metrics.items():
            metric_results.append(MetricResult(name=key, value=value))

        self._log_value = {
            eval_def.TIMESTAMP_KEY_NAME: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            eval_def.STEP_KEY_NAME: step,
            eval_def.METRICS_KEY_NAME: metric_results
        }

    def add_histogram(self,
                      tag: str,
                      values: torch.Tensor,
                      global_step: int = None,
                      bins: str = "auto",
                      walltime: Typing.Optional[Typing.Union[float, None]] = None) -> None:
        """
        Adds a histogram to the logger.

        Parameters:
        ----------
        tag: str
            The name of the histogram.
        values: torch.Tensor
            The bincounts for the histogram.
        step: int, optional
            The step at which the histogram is logged.
        bins: int, optional
            The number of bins for the histogram.

        Returns:
        ----------
        None
        """
        if self._histogram is not None:
            self._histogram[tag] = values
        else:
            self._histogram = {tag: values}

    def get_last_log(self) -> Typing.Dict[str, Typing.Any]:
        """
        Returns the last log entry.

        Parameters:
        ----------
        None

        Returns:
        ----------
        dict: The last log entry.
        """
        return self._log_value

    def get_histogram(self) -> Typing.Union[CodebookUsage, None]:
        """
        Returns the last histogram entry.

        Parameters:
        ----------
        None

        Returns:
        ----------
        dict: The last histogram entry.
        """
        if not hasattr(self, "_histogram"):
            return None

        codebook_statistics: CodebookUsage = []
        for key, value in self._histogram.items():
            # tag is tag=f"{self.tag_name}_space_{codebook_num}",
            codebook_num = int(key.split("_space_")[-1])
            codebook_statistics.append(CodebookUsageEntry(codebook_index=codebook_num, bincount=value))

        return codebook_statistics

    def reset_histogram(self) -> None:
        """
        Resets the histogram data.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
        """
        if hasattr(self, "_histogram"):
            self._histogram = None
