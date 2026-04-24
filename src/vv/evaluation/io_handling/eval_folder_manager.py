import os
import pickle
import re
import shutil
import typing as Typing
from typing import Tuple

from filelock import FileLock

import vv.evaluation.core.definitions as eval_def
from vv.evaluation.core.core import CheckpointInfo
from vv.evaluation.io_handling.info_file_handler import InfoFileHandler


class EvalFolderManager:
    """
    A class to manage eval directory for model checkpoints, including retrieving their paths
    and replicating the directory hierarchy in a destination folder.
    """
    def __init__(self,
                 source_path: str,
                 destination_path: str,
                 overwrite_possible: bool) -> None:
        """
        Initializes the manager with source and destination paths.

        Parameters:
        -----------
        source_path: str
            The path to the source folder or file.
        destination_path: str
            The path to the destination folder.
        overwrite_possible: bool
            Flag indicating whether to overwrite existing results is prompted.
        """
        self.source_path = os.path.abspath(source_path)
        self.destination_path = os.path.join(os.path.abspath(destination_path), eval_def.EVAL_FOLDER_NAME)
        self.overwrite_possible = overwrite_possible

        self.dir_structure_exists = False
        self.info_file_handler = InfoFileHandler(self.destination_path)

    def create_basic_folder_structure(self) -> None:
        """
        Creates the basic folder structure for the output files.
        If the destination path already exists, it prompts the user to override it.
        Otherwise, it creates the directory.

        Parameters:
        -----------
        None

        Returns:
        -----------
        dict: Information on the created directories and checkpoint names.
        """
        if os.path.exists(self.destination_path):
            folder_created = self._handle_dir_exists()
        else:
            folder_created = self._handle_dir_not_exists()

        if folder_created:
            self.info_file_handler.create_info_file()
            models = self._make_model_list()
            self.info_file_handler.add_model(models)
        else:
            self.info_file_handler.get_info_file()
            models = self._make_model_list()
            models = self.info_file_handler.update_model_list(models, self.overwrite_possible)

        self._create_directories(models)

    def add_results(self,
                    step: int,
                    checkpoint_path: str,
                    csv_formatted_result: Tuple[str, str],
                    objs: Typing.Optional[Typing.Dict[str, Typing.Any]]) -> None:
        """
        Adds the result to the results.txt file in the corresponding eval directory.
        And updates the info file with the processed model.

        Parameters:
        -----------
        step: int
            The training step of the model.
        checkpoint_path: str
            The path to the checkpoint file.
        csv_formatted_result: Tuple[str, str]
            A tuple containing the CSV header and the CSV row string.
        objs: dict, optional
            A dictionary of objects that will be added besides the results.
            The keys are the names of the objects, and the values are the objects themselves.

        Returns:
        -----------
        None
        """
        if not self.dir_structure_exists:
            raise RuntimeError("Folder Manager not properly initialized."
                               "Please call create_basic_folder_structure() first.")

        self._write_results(checkpoint_path, csv_formatted_result)
        self._add_objects_to_results(step, checkpoint_path, objs)

        # Mark as processed
        self.info_file_handler.mark_as_processed(checkpoint_path)

    def add_to_failed_models(self,
                             checkpoint_path: str) -> None:
        """
        Adds the checkpoint path to the failed models list in the info file.

        Parameters:
        -----------
        checkpoint_path: str
            The path to the checkpoint file.

        Returns:
        -----------
        None
        """
        self.info_file_handler.mark_as_failed(checkpoint_path)

    def get_next_model(self) -> CheckpointInfo:
        """
        Retrieves the next model to process from the info file.

        Returns:
        -----------
        str: The path to the next model to process.
        """
        if not self.dir_structure_exists:
            raise RuntimeError("Folder Manager not properly initialized."
                               "Please call create_basic_folder_structure() first.")

        ckpt_path = self.info_file_handler.get_next_model_to_process()
        ckpt_config_path = self._get_config_path(ckpt_path)

        return CheckpointInfo(ckpt_path=ckpt_path, config_path=ckpt_config_path)

    def _write_results(self,
                       ckpt_path: str,
                       csv_formatted_result: Tuple[str, str]) -> None:
        """
        Writes the result to the results.txt file in the specified eval directory.
        Adds the header if it does not exist.

        Parameters:
        -----------
        ckpt_path: str
            The path to the checkpoint file.
        csv_formatted_result: Tuple[str, str]
            A tuple containing the CSV header and the CSV row string.

        Returns:
        -----------
        None
        """
        header, row = csv_formatted_result
        eval_dir = self._get_eval_dir_from_ckpt_path(ckpt_path)

        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        results_file_path = os.path.join(eval_dir, eval_def.RESULTS_FILE_NAME)

        with FileLock(f"{results_file_path}.lock"):
            # Read existing content if the file exists
            file_exists = os.path.exists(results_file_path)
            header_exists = False

            if file_exists:
                with open(results_file_path, "r") as results_file:
                    first_line = results_file.readline().strip()
                    header_exists = first_line == header

            # Write to the file (append mode)
            with open(results_file_path, "a") as results_file:
                if not header_exists:
                    results_file.write(f"{header}{os.linesep}")
                results_file.write(f"{row}{os.linesep}")

    def _add_objects_to_results(self,
                                step: int,
                                ckpt_path: str,
                                objs: Typing.Optional[Typing.Dict[str, Typing.Any]]) -> None:
        """
        Adds the objects to the results file in the specified eval directory.
        Parameters:
        -----------
        step: int
            The training step of the model.
        ckpt_path: str
            The path to the checkpoint file.
        objs: dict, optional
            A dictionary of objects to add to the results file.
            The keys are the names of the objects, and the values are the objects themselves.
        Returns:
        -----------
        None
        """
        if objs is None:
            return
        eval_dir = self._get_eval_dir_from_ckpt_path(ckpt_path)

        for obj_name, obj_value in objs.items():
            obj_dir = os.path.join(eval_dir, obj_name)
            if not os.path.exists(obj_dir):
                os.makedirs(obj_dir)

            obj_file_path = os.path.join(obj_dir, f"{step}.pkl")
            with open(obj_file_path, "wb") as obj_file:
                pickle.dump(obj_value, obj_file)

    def _get_eval_dir_from_ckpt_path(self,
                                     ckpt_path: str) -> str:
        """
        Extracts the eval directory from the checkpoint path.

        Parameters:
        -----------
        ckpt_path: str
            The path to the checkpoint file.

        Returns:
        -----------
        str: The eval directory corresponding to the checkpoint path.
        """
        parts = ckpt_path.split(os.sep)

        # Find the version_x part and its index
        version_idx = next((i for i, p in enumerate(parts) if re.match(r'version_\d+', p)), None)
        if version_idx is None:
            raise ValueError(f"No valid version_x segment in path: {ckpt_path}")

        model_name = parts[version_idx - 1]
        eval_dir = os.path.join(self.destination_path, model_name)

        return eval_dir

    def _handle_dir_exists(self) -> bool:
        """
        Handles the case where the destination path already exists.
        Prompts the user to either override the existing directory or exit.

        Parameters:
        -----------
        None

        Returns:
        -----------
        bool: True if the directory was overridden, False otherwise.
        """
        if self.overwrite_possible:
            user_input = input(f"The directory '{self.destination_path}' already exists. Do you want to override it? (yes/no): ").strip().lower()
            if user_input in ["yes", "y"] and input("Are you sure? This will delete all existing files (yes/no): ").strip().lower() in ["yes", "y"]:
                print(f"Overriding the existing directory.{os.linesep}")
                shutil.rmtree(self.destination_path)
                os.makedirs(self.destination_path)
                return True
            print("Continuing in existing directory.")
            return False
        else:
            print("Continuing in existing directory.")
            return False

    def _handle_dir_not_exists(self) -> bool:
        """
        Handles the case where the destination path does not exist.
        Creates the directory at the specified path.

        Parameters:
        -----------
        None

        Returns:
        -----------
        bool: True if the directory was created successfully.
        """
        os.makedirs(self.destination_path)

        return True

    def _make_model_list(self) -> Typing.List[str]:
        """
        Recursively scans the source directory for checkpoint files (.ckpt).

        Returns:
        -----------
        List[str]: A list of absolute paths to all checkpoint files found.
        """
        checkpoint_files: Typing.List[str] = []

        for root, _, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(".ckpt"):
                    full_file_name = os.path.abspath(os.path.join(root, file))
                    checkpoint_files.append(full_file_name)

        return checkpoint_files

    def _check_valid_path_property(self,
                                   path: str) -> None:
        """
        Checks if the given path is a valid directory.

        Parameters:
        -----------
        path: str
            The path to check.

        Returns:
        -----------
        None
        """
        if os.path.isfile(path):
            raise ValueError("Source path must be a folder.")

    def _get_config_path(self,
                         ckpt_path: Typing.Union[str, None]) -> Typing.Union[str, None]:
        """
        Retrieves the configuration path from the checkpoint path.

        Parameters:
        -----------
        ckpt_path: str
            The path to the checkpoint file.

        Returns:
        -----------
        str: The configuration path corresponding to the checkpoint path.
        """
        if ckpt_path is None:
            return None

        parts = ckpt_path.split(os.sep)

        # Find the version_x part and its index
        version_idx = next((i for i, p in enumerate(parts) if re.match(r'version_\d+', p)), None)
        if version_idx is None:
            raise ValueError(f"No valid version_x segment in path: {ckpt_path}")

        config_path = os.path.join("/", *parts[0:version_idx + 1], eval_def.CONFIG_FILE_NAME)

        return config_path

    def _create_directories(self,
                            model_ckpts: Typing.List[str]) -> None:
        """
        Creates the eval directories based on the list of model checkpoints.

        Parameters:
        -----------
        model_ckpts: List[str]
            A list of model checkpoint paths.

        Returns:
        -----------
        None
        """
        for model_ckpt in model_ckpts:
            eval_dir = self._get_eval_dir_from_ckpt_path(model_ckpt)
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

        self.dir_structure_exists = True
