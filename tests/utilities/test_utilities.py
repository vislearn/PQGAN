import os
import shutil
import unittest

import vv.utilities as utils

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No cuda on Github.")


class TestUtilities(unittest.TestCase):
    """
    Test class for the utilities functions
    """
    def test_path(self) -> None:
        """
        Test the get_dir function
        """

        path = utils.get_dir()

        # Assert that the path exists
        self.assertTrue(os.path.exists(path))

        # Get a list of all directories and files in path
        contents = os.listdir(path)

        # Assert that 'src' is in the list of directories/files
        self.assertTrue('src' in contents)

    def test_delete_file(self) -> None:
        """
        Test the delete_file function
        """

        # Create a folder for a file
        folder_path = utils.get_dir('__tmp__', create=True)
        # Create the file
        with open(os.path.join(folder_path, 'test_file.txt'), 'w') as file:
            file.write('This is an automatically generated test file')

        file_path = os.path.join(folder_path, 'test_file.txt')

        # Assert that the file exists
        self.assertTrue(os.path.exists(file_path))

        # Delete the file
        utils.delete_file(file_path)

        # Assert that the file no longer exists
        self.assertFalse(os.path.exists(file_path))

        # Delete the folder
        shutil.rmtree(folder_path)
