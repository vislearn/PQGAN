import os
import traceback
from typing import Tuple

import torch

from vv.evaluation.core.core import CheckpointInfo
from vv.evaluation.core.result_data import ExperimentResult
from vv.evaluation.io_handling import EvalFolderManager
from vv.evaluation.post_processing.formatter import Formatter
from vv.evaluation.runtime.model_evaluator import ModelEvaluator


class EvalExecutionManager:
    """
    Manages the evaluation process by coordinating the EvalFolderManager and ModelEvaluator.
    """

    def __init__(self,
                 source_path: str,
                 destination_path: str,
                 eval_config_path: str,
                 overwrite_possible: bool) -> None:
        """
        Initializes the EvalExecutionManager.

        Parameters:
        -----------
        source_path: str
            The path to the source folder containing model checkpoints.
        destination_path: str
            The path to the destination folder for evaluation results.
        eval_config_path: str
            The path to the evaluation configuration file.
        overwrite_possible: bool
            Flag indicating whether to overwrite existing results is prompted.
        """
        torch.set_float32_matmul_precision('medium')

        self.source_path = os.path.abspath(source_path)
        self.destination_path = os.path.abspath(destination_path)
        self.eval_config_path = os.path.abspath(eval_config_path)

        self.folder_manager = EvalFolderManager(self.source_path, self.destination_path, overwrite_possible)
        self.model_evaluator = ModelEvaluator(self.eval_config_path)
        self.formatter = Formatter()

        self.failed_streak = 0
        self.batch_size_reduction = 0.75
        self.batch_size_reduction_max_attempts = 5
        self.current_failed_streak = 0
        self.max_failed_streak = 3

    def run_evaluation(self) -> None:
        """
        Runs the evaluation process for all models.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        """
        self.folder_manager.create_basic_folder_structure()
        success = self._process_all_models()

        if success:
            print("All models have been processed.")
        else:
            print("Evaluation process was interrupted due to too many failures.")

    def _process_all_models(self) -> bool:
        """
        Iterates through all models and evaluates them.

        Parameters:
        -----------
        None

        Returns:
        -----------
        bool
            True if all models were processed successfully, False otherwise.
        """
        success = True
        while (ckpt_info := self.folder_manager.get_next_model()).ckpt_path is not None:
            print(f"Evaluating model: {ckpt_info.ckpt_path}")
            self._evaluate_with_retries(ckpt_info)

            if self.current_failed_streak >= self.max_failed_streak:
                success = False
                break

        return success

    def _evaluate_with_retries(self,
                               ckpt_info: CheckpointInfo) -> None:
        """
        Evaluates a model with retries on CUDA OOM errors.

        Parameters:
        -----------
        ckpt_info : CheckpointInfo
            Information about the model checkpoint to evaluate.
        """
        current_batch_size_reduction = 1.0

        for attempt in range(self.batch_size_reduction_max_attempts):
            try:
                result: ExperimentResult = self.model_evaluator.evaluate_model(ckpt_info,
                                                                               current_batch_size_reduction)
                self._save_evaluation_results(ckpt_info, result)
                break

            except Exception as e:
                self.model_evaluator = ModelEvaluator(self.eval_config_path)
                if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e) and attempt < self.batch_size_reduction_max_attempts - 1:
                    current_batch_size_reduction *= self.batch_size_reduction
                    print("CUDA OOM encountered")
                    print(f"Reducing original batch size by {current_batch_size_reduction} and retrying...")
                    print(f"Attempt {attempt+1}/{self.batch_size_reduction_max_attempts}")
                else:
                    self._handle_nonrecoverable_error(ckpt_info)
                    break

    def _handle_nonrecoverable_error(self,
                                     ckpt_info: CheckpointInfo) -> None:
        """
        Handles logging and state update for non-recoverable errors.

        Parameters:
        -----------
        ckpt_info : CheckpointInfo
            Information about the model checkpoint that caused the error.
        """
        print(f"Non-recoverable error while evaluating model {ckpt_info.ckpt_path}")
        self.folder_manager.add_to_failed_models(ckpt_info.ckpt_path)
        traceback.print_exc()
        self.current_failed_streak += 1

    def _save_evaluation_results(self,
                                 ckpt_info: CheckpointInfo,
                                 result: ExperimentResult) -> None:
        """
        Saves evaluation results to disk.

        Parameters:
        -----------
        ckpt_info : CheckpointInfo
            Information about the model checkpoint.
        result : dict
            The result returned by the evaluator.

        Returns:
        -----------
        None
        """
        step = result.step
        csv_formatted_result: Tuple[str, str] = self.formatter.to_csv(result)
        additional_result_objs = self.formatter.to_obj_dict(result)

        self.folder_manager.add_results(step, ckpt_info.ckpt_path, csv_formatted_result, additional_result_objs)

        print(f"Successfully evaluated and saved results for model: {ckpt_info.ckpt_path}")
        self.current_failed_streak = 0
