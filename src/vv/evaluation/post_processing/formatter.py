import typing as Typing
from dataclasses import fields, is_dataclass

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.core.result_data import ExperimentResult, MetricResult


class Formatter:
    def __init__(self) -> None:
        return

    def to_obj_dict(self,
                    result: ExperimentResult) -> Typing.Dict[str, Typing.Any]:
        """
        Convert the ExperimentResult object into a dict of objects dynamically.

        Parameters
        ----------
        log_value : ExperimentResult
            An ExperimentResult object containing the attributes to log.

        Returns
        -------
        dict
            A dictionary representation of additional metrics in the ExperimentResult.
        """
        try:
            # Convert the dataclass to a dictionary
            result_dict = {}
            for field in fields(result):
                value = getattr(result, field.name)
                # If the field is a dataclass, keep it as-is
                if is_dataclass(value):
                    result_dict[field.name] = value
                else:
                    result_dict[field.name] = value

            # Remove unwanted keys
            keys_to_remove = {"experiment_id", "timestamp", "step", "metrics", "metadata"}
            result_dict = {k: v for k, v in result_dict.items() if k not in keys_to_remove}

            # Remove keys with empty or default-initialized values
            result_dict = {k: v for k, v in result_dict.items() if v not in (None, "", [], {}, ())}

            return result_dict
        except Exception as e:
            print(f"Error while converting ExperimentResult to dict: {e}")
            return {}

    def to_csv(self,
               results: ExperimentResult) -> Typing.Tuple[str, str]:
        """
        Convert the metrics in the ExperimentResult to a CSV compatible string (header and row).

        Parameters
        ----------
        results : ExperimentResult
            An ExperimentResult object containing all relevant result information.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the CSV header and the CSV row string.
        """
        try:
            header = self._get_csv_header(results.metrics)

            timestamp = results.timestamp
            step = results.step
            metrics = results.metrics

            values = [timestamp, str(step)] + [str(metric.value) for metric in metrics]

            row = ",".join(values)
        except Exception as e:
            print(f"Error while converting results to csv compatible string: {e}")
            header = ""
            row = ""

        return header, row

    def _get_csv_header(self,
                        metrics: MetricResult) -> str:
        """
        Generate the CSV header from the log value dictionary.

        Parameters
        ----------
        log_value: MetricResult
            A MetricResult object containing the metrics to log.

        Returns
        -------
        str
            A CSV header string.
        """
        metric_names = [metric.name for metric in metrics]
        header = [eval_def.TIMESTAMP_KEY_NAME, eval_def.STEP_KEY_NAME] + metric_names
        return ",".join(header)
