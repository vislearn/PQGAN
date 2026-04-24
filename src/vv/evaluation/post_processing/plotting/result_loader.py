import os
import pickle
import typing as Typing

import pandas as pd
from tqdm import tqdm

import vv.evaluation.core.definitions as eval_def


class EvaluationResultsLoader:
    """
    A class to load CSV files and convert them into a list of dictionaries.
    """

    def __init__(self,
                 destination_path: str) -> None:
        self.destination_path_eval = os.path.abspath(destination_path)

    def load_results(self) -> Typing.Dict[str, pd.DataFrame]:
        """
        Load results from the destination path and check for NaN values.

        Parameters
        ----------
        None

        Returns
        ----------
        dict:
            A dictionary where keys are model names and values are DataFrames containing the results.
        """
        model_folders = [
            model_folder for model_folder in os.scandir(self.destination_path_eval)
            if model_folder.is_dir() and model_folder.name != eval_def.PRCESSED_RESULTS_FOLDER_NAME
        ]
        data: Typing.Dict[str, pd.DataFrame] = {}

        for model_folder in tqdm(model_folders, desc="Loading data", unit="folder"):
            # Load results from the result CSV file
            df_csv = self._load_results_from_file(model_folder.path)

            # Load results from the additional eval objects
            df_obj = self._load_results_from_object(model_folder.path)

            # Merge the two dataframes over steps
            df = None
            if df_csv is not None and df_obj is not None:
                df = pd.merge(df_csv, df_obj, on=eval_def.STEP_KEY_NAME, how='outer')
            elif df_csv is not None:
                df = df_csv
            elif df_obj is not None:
                df = df_obj

            if df is not None:
                data[model_folder.name] = df
            else:
                tqdm.write(f"Warning: {eval_def.RESULTS_FILE_NAME} not found for {model_folder.path}")

        return data

    def _load_results_from_file(self,
                                model_path: str) -> Typing.Union[pd.DataFrame, None]:
        """
        Load results from a specific model path. Results are expected to be in a CSV file.

        Parameters
        ----------
        model_path : str
            Path to the model folder.

        Returns
        ----------
        df: pd.DataFrame or None
            DataFrame containing the results, or None if the file does not exist.
        """
        df = None

        # Read the entries in the result file and add them to the dataframe
        result_file = os.path.join(model_path, eval_def.RESULTS_FILE_NAME)
        if os.path.exists(result_file):
            try:
                df = pd.read_csv(result_file, delimiter=",")
                self._detect_invalid_entires(df)
            except Exception as e:
                raise RuntimeError(f"Error loading {result_file}") from e

        return df

    def _load_results_from_object(self,
                                  model_path: str) -> Typing.Union[pd.DataFrame, None]:

        """
        Load results from a specific model path where results are stored as pickle files.

        Parameters
        ----------
        model_path : str
            Path to the model folder.

        Returns
        ----------
        df : pd.DataFrame or None
            DataFrame containing the results, or None if no valid pickle files are found.
        """
        df = None

        # Scan for pickle files in subdirectories
        additional_eval_objects_paths = self._get_additional_eval_object_paths(model_path)

        # Collect data from pickle files
        data: dict = {}
        steps: set = set()
        for obj_path in additional_eval_objects_paths:
            obj_name = os.path.basename(obj_path)
            data[obj_name] = {}
            for file_name in os.listdir(obj_path):
                file_path = os.path.join(obj_path, file_name)
                obj, step = self._get_obj_from_file(file_path)
                if obj is not None:
                    data[obj_name][step] = obj
                    steps.add(step)

        # Create DataFrame
        if data:
            sorted_steps = sorted(steps)  # Convert the set to a sorted list
            df_data = {obj_name: [data[obj_name].get(step, None) for step in sorted_steps] for obj_name in data}
            df_data[eval_def.STEP_KEY_NAME] = sorted_steps
            df = pd.DataFrame(df_data)

        return df

    def _detect_invalid_entires(self,
                                df: pd.DataFrame) -> None:
        """
        Detect invalid entries in the DataFrame.
        This method checks for NaN values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check for invalid entries.

        Returns
        ----------
        None
        """
        if df.isnull().values.any():
            print("Warning: NaN detected in the DataFrame.")

    def _get_additional_eval_object_paths(self,
                                          model_path: str) -> Typing.List[str]:
        """
        Get additional object names from the model path.

        Parameters
        ----------
        model_path : str
            Path to the model folder.

        Returns
        ----------
        Typing.List[str]
            List of additional object names found in the model path.
        """
        additional_eval_objects = [os.path.abspath(d.path) for d in os.scandir(model_path) if d.is_dir()]
        if not additional_eval_objects:
            return []

        return additional_eval_objects

    def _get_obj_from_file(self,
                           file: str) -> Typing.Tuple[Typing.Union[Typing.Any, None], int]:
        """
        Get the object from the file.
        This method reads the file and returns the object and its step.

        Parameters
        ----------
        file : str
            Path to the file.

        Returns
        ----------
        Typing.Tuple[Typing.Union[Typing.Any, None], int]
            Tuple containing the object and its step.
        """
        obj = None
        step = -1

        if os.path.isfile(file) and file.endswith(".pkl"):
            try:
                step = int(os.path.splitext(os.path.basename(file))[0])
                with open(file, "rb") as f:
                    obj = pickle.load(f)  # Deserialize the object from the pickle file
            except Exception as e:
                raise RuntimeError(f"Error loading pickle file {file}") from e

        return obj, step
