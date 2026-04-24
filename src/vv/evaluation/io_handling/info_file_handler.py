import os
import typing as Typing

from filelock import FileLock


class InfoFileHandler:
    """
    Handles the output files for storing model information.

    Attributes:
    -----------
    destination_path: str
        The path where the output file will be created.
    info_file_path: str
        The path to the output file.
    """

    info_file_name: str = "info.txt"
    section_all_models: str = "All Models"
    section_processed_models: str = "Processed Models"
    section_models_in_processing: str = "Models in Processing"
    section_models_to_process: str = "Models to Process"
    section_failed_models: str = "Models Failed to Process"

    section_order: dict = {
        section_all_models: 0,
        section_processed_models: 1,
        section_models_in_processing: 2,
        section_models_to_process: 3,
        section_failed_models: 4,
        "EOF": 5
    }

    def __init__(self,
                 destination_path: str):
        """
        Initializes the OutputFileHandler with the destination path.

        Parameters:
        -----------
        destination_path: str
            The path to the destination folder or file.
        """
        self.destination_path = os.path.abspath(destination_path)
        self.info_file_path = ""

    def create_info_file(self) -> None:
        """
        Creates the info file in the destination path.
        Initializes the file with sections for all models, processed models, and models to process.
        If the file already exists, it prompts the user to override it.

        Parameters:
        -----------
        None

        Returns:
        -----------
        str: The path to the created info file.
        """
        info_file_path = os.path.join(self.destination_path, InfoFileHandler.info_file_name)

        with open(info_file_path, "w") as file:
            for section in InfoFileHandler.section_order.keys():
                if section != "EOF":  # Skip the EOF marker
                    file.write(f"{section}:{os.linesep}")
                    file.write(os.linesep)

        self._set_filename(info_file_path)

    def get_info_file(self) -> None:
        """
        Returns the path to the existing info file.

        Parameters:
        -----------
        None

        Returns:
        -----------
        str: The path to the existing info file.
        """
        info_file_path = os.path.join(self.destination_path, InfoFileHandler.info_file_name)

        try:
            with open(info_file_path, "r") as _:
                pass
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Info file not found at {self.info_file_path}. "
                "Please ensure the file exists or delete the directory to create a new one."
            ) from e

        self._set_filename(info_file_path)

    def add_model(self,
                  model_path: Typing.Union[str, list[str]]) -> None:
        """
        Adds a model(s) to the 'All Models' and 'Models to Process' sections.

        Parameters:
        -----------
        model_path: str
            The path to the model file.
        """
        if type(model_path) is str:
            self._add_single_model(model_path)
        elif type(model_path) is list:
            for model in model_path:
                self._add_single_model(model)

    def mark_as_in_processing(self,
                              model_path: str) -> None:
        """
        Moves a model from 'Models to Process' to 'Models in Processing'.

        Parameters:
        -----------
        model_path: str
            The path to the model file.
        """
        with self.lock:
            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Remove the model from the "Models to Process" section
                line_index = self._get_model_index(model_path,
                                                   InfoFileHandler.section_models_to_process,
                                                   content)

                content[line_index] = ""

                file.seek(0)
                file.writelines(content)
                file.truncate()

            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Add to "Models in Processing" section
                start_index, _ = self._get_section_index(InfoFileHandler.section_models_in_processing, content)
                content.insert(start_index + 1, f"{model_path}{os.linesep}")

                file.seek(0)
                file.writelines(content)

    def mark_as_processed(self,
                          model_path: str) -> None:
        """
        Moves a model from 'Models to Process' to 'Processed Models'.

        Parameters:
        -----------
        model_path: str
            The path to the model file.
        """
        with self.lock:
            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Remove the model from the "Models to Process" section
                line_index = self._get_model_index(model_path,
                                                   InfoFileHandler.section_models_in_processing,
                                                   content)

                content[line_index] = ""

                file.seek(0)
                file.writelines(content)
                file.truncate()

            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Add to "Processed Models" section
                start_index, _ = self._get_section_index(InfoFileHandler.section_processed_models, content)
                content.insert(start_index + 1, f"{model_path}{os.linesep}")

                file.seek(0)
                file.writelines(content)

    def mark_as_failed(self,
                       model_path: str) -> None:
        """
        Moves a model from 'Models to Process' to 'Models Failed to Process'.

        Parameters:
        -----------
        model_path: str
            The path to the model file.
        """
        with self.lock:
            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Remove the model from the "Models to Process" section
                line_index = self._get_model_index(model_path,
                                                   InfoFileHandler.section_models_in_processing,
                                                   content)

                content[line_index] = ""

                file.seek(0)
                file.writelines(content)
                file.truncate()

            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Add to "Models Failed to Process" section
                start_index, _ = self._get_section_index(InfoFileHandler.section_failed_models, content)
                content.insert(start_index + 1, f"{model_path}{os.linesep}")

                file.seek(0)
                file.writelines(content)
                file.truncate()

    def get_next_model_to_process(self) -> Typing.Union[str, None]:
        """
        Retrieves the next model to process from the 'Models to Process' section.

        Returns:
        -----------
        str: The path to the next model to process.
        """
        with self.lock:
            with open(self.info_file_path, "r") as file:
                content = file.readlines()

            model_path = None
            start_index, end_index = self._get_section_index(InfoFileHandler.section_models_to_process, content)
            for line in content[start_index + 1:end_index]:
                stripped_line = line.strip()
                if stripped_line:  # Check if the line is not empty
                    self.mark_as_in_processing(stripped_line)
                    model_path = stripped_line
                    break

        return model_path

    def get_all_models(self) -> Typing.List[str]:
        """
        Retrieves the list of all models stored in the info file.

        Returns:
        -----------
        List[str]: A list of all model paths.
        """
        with self.lock:
            with open(self.info_file_path, "r") as file:
                content = file.readlines()

        start_index, end_index = self._get_section_index(InfoFileHandler.section_all_models, content)
        return [line.strip() for line in content[start_index + 1:end_index] if line.strip()]

    def get_models_to_process(self) -> Typing.List[str]:
        """
        Retrieves the list of models that still need to be processed.

        Returns:
        -----------
        List[str]: A list of model paths that are yet to be processed.
        """
        with self.lock:
            with open(self.info_file_path, "r") as file:
                content = file.readlines()

        start_index, end_index = self._get_section_index(InfoFileHandler.section_models_to_process, content)
        return [line.strip() for line in content[start_index + 1:end_index] if line.strip()]

    def update_model_list(self,
                          model_path_new: Typing.Union[str, list[str]],
                          overwrite_possible: bool) -> Typing.Union[str, list[str]]:
        """
        Updates the model list in the info file.

        Parameters:
        -----------
        model_path_new: str
            The path to the model file.

        Returns:
        -----------
        List[str]: A list of all model paths, including any new models added.
        """
        model_list_old = self.get_all_models()

        if not overwrite_possible:
            print("Did not search for new modles because overwrite_possible is set to False.")
            return model_list_old

        # Compare the old and new model lists
        new_models = [model for model in model_path_new if model not in model_list_old]

        if new_models:
            print("The following models were found that are not contained in the info file:")
            for model in new_models:
                print(f"- {model}")

            user_input = input("Do you want to add these models to the info file? (yes/no): ").strip().lower()
            if user_input in ["yes", "y"]:
                self.add_model(new_models)
                print("New models have been added to the info file.")
                return model_list_old + new_models
            else:
                print("No new models were added. Continuing with the existing list.")
                return model_list_old
        else:
            print("No new models found. Continuing with the existing list.")
            return model_list_old

    def _get_section_index(self,
                           section_title: str,
                           content: Typing.List[str]) -> Typing.Tuple[int, int]:
        """
        Returns the start and end indices of a section in the info file.

        Parameters:
        -----------
        section_title: str
            The title of the section to find.
        content: str
            The content of the info file.

        Returns:
        -----------
        Tuple[int, int]: The start and end indices of the section in the file.
        """
        # Extract the full section name
        full_section_name = f"{section_title}:{os.linesep}"
        section_number = InfoFileHandler.section_order.get(section_title)

        if section_number is None:
            raise RuntimeError(f"Section header '{section_title}' is not a valid section header.")

        # Determine the next section name
        next_section_number = section_number + 1
        if next_section_number < len(InfoFileHandler.section_order):
            next_section_title = list(InfoFileHandler.section_order.keys())[next_section_number]
            next_full_section_name = f"{next_section_title}:{os.linesep}"
        else:
            raise RuntimeError(f"Section header '{section_title}' is the last section header.")

        try:
            # Find the index of the section title
            start_index = content.index(full_section_name)
        except ValueError as e:
            raise RuntimeError(
                f"File compromised: Section header '{full_section_name.strip()}' not found."
            ) from e

        try:
            if next_full_section_name == f"EOF:{os.linesep}":
                end_index = -1
            else:
                end_index = content.index(next_full_section_name) - 1
        except ValueError as e:
            raise RuntimeError(
                f"File compromised: Section header '{next_full_section_name.strip()}' not found."
            ) from e

        return start_index, end_index

    def _get_model_index(self,
                         model_path: str,
                         section_title: str,
                         content: Typing.List[str]) -> int:
        """
        Returns the start and end indices of a section in the info file.

        Parameters:
        -----------
        model_path: str
            The path of the model to find.
        section_title: str
            The title of the section to find.
        next_section_title: str
            The title of the next section to find.
        content: str
            The content of the info file.

        Returns:
        -----------
        int: The index of the section title in the file.
        """
        # Locate the section start and end indices
        section_start, section_end = self._get_section_index(section_title, content)

        # Limit the search to the section's content
        section_content = content[section_start + 1:section_end]

        # Search for the model within the section
        full_name = f"{model_path}{os.linesep}"
        try:
            index_in_section = section_content.index(full_name)
            return section_start + 1 + index_in_section  # Adjust index relative to the full content
        except ValueError as e:
            raise RuntimeError(
                f"Model '{model_path}' not found in section '{section_title}'."
            ) from e

    def _add_single_model(self,
                          model_path: str) -> None:
        """
        Adds a single model to the 'All Models' and 'Models to Process' sections.

        Parameters:
        -----------
        model_path: str
            The path to the model file.
        """
        with self.lock:
            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Add to "All Models" section
                start_index, _ = self._get_section_index(InfoFileHandler.section_all_models, content)
                content.insert(start_index + 1, f"{model_path}{os.linesep}")

                file.seek(0)
                file.writelines(content)

            with open(self.info_file_path, "r+") as file:
                content = file.readlines()

                # Add to "Models to Process" section
                start_index, _ = self._get_section_index(InfoFileHandler.section_models_to_process, content)
                content.insert(start_index + 1, f"{model_path}{os.linesep}")

                file.seek(0)
                file.writelines(content)

    def _set_filename(self,
                      filename: str) -> None:
        """
        Sets the filename for the info file.

        Parameters:
        -----------
        filename: str
            The name of the info file.
        """
        self.info_file_path = filename
        self.lock = FileLock(f"{self.info_file_path}.lock")
