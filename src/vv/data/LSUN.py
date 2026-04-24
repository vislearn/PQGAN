import os
import typing as Typing

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vv.data.core.data_class import CoreDataModule

ClassesInputType = Typing.Union[str, Typing.List[str]]


class LSUNLoader(CoreDataModule):
    """
    DataModule for LSUN dataset.
    Supports loading specific categories or all.

    10 categories are available.
    In Train each category has 120,000 to 3,000,000 samples.
    In Val each category has 300 samples. --> 3,000 samples in total.
    In Test each category has 1,000 samples. --> 10,000 samples in total.

    Classes are:
     - bedroom
     - bridge
     - church_outdoor
     - classroom
     - conference_room
     - dining_room
     - kitchen
     - living_room
     - restaurant
     - tower
    """

    def __init__(self,
                 resolution: int = 256,
                 train_classes: ClassesInputType = "train",
                 val_classes: ClassesInputType = "val",
                 test_classes: ClassesInputType = "test",
                 norm_values: Typing.Optional[Typing.Tuple[float, float]] = None,
                 batch_size: int = 128,
                 workers: int = 10,
                 verbose: bool = False,
                 data_dir: str = "/path/to/lsun") -> None:
        """
        Parameters:
        ----------
            resolution: int
                Resolution of the images.
            train_classes str or list: One of {'train', 'val', 'test'} or a list of
                categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
                This is used for training.
            val_classes str or list: One of {'train', 'val', 'test'} or a list of
                categories to load. e,g. ['bedroom_val', 'church_outdoor_val'].
                This is used for validation.
            test_classes str or list: One of {'train', 'val', 'test'} or a list of
                categories to load. e,g. ['bedroom_test', 'church_outdoor_test'].
                This is used for testing.
            norm_values: Typing.Optional[Typing.Tuple[float, float]]
                Values for the data normalization.
                First value is the mean, second value is the std.
                If None, the default values are used.
                Hint: 0.5 0.5 for a range of -1 to 1
                Hint 0.0 1.0 for a range of 0 to 1
            batch_size: int
                Batch size for the DataLoader.
            workers: int
                Number of workers for the DataLoader.
            verbose: bool
                Whether to print out information.
            data_dir: str
                Directory where the data is stored.
        """
        super().__init__()
        self.resolution = resolution
        self.train_classes = train_classes
        self.val_classes = val_classes
        self.test_classes = test_classes
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose
        self.data_dir = data_dir

        if norm_values is None:
            self.normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])
        else:
            self.normalization = transforms.Normalize(mean=[norm_values[0], norm_values[0], norm_values[0]],
                                                      std=[norm_values[1], norm_values[1], norm_values[1]])

        self.transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            self.normalization
        ])

        self.save_hyperparameters()

    def get_normalization(self) -> transforms.Normalize:
        """
        Returns the normalization transform used for the dataset.

        Returns:
        ----------
            transforms.Normalize: A transform object that normalizes the data.
        """
        return self.normalization

    def prepare_data(self) -> None:
        """
        Prepares the data by checking if the LSUN directory exists.
        """
        assert os.path.exists(self.data_dir), f"LSUN directory not found at {self.data_dir}"

    def setup(self,
              stage: str) -> None:
        """
        Sets up the dataset for training, validation, and testing.

        Parameters:
        ----------
            stage: str
                The stage of the data module (fit, validate, test, predict).
        """
        if stage == "fit":
            # Check if train_classes and val_classes are provided
            self._verify_valid_classes_input_parameter(self.train_classes)
            self._verify_valid_classes_input_parameter(self.val_classes)

            # Create datasets for training and validation
            self.dataset_train = datasets.LSUN(root=self.data_dir,
                                               classes=self.train_classes,
                                               transform=self.transform)
            self.dataset_val = datasets.LSUN(root=self.data_dir,
                                             classes=self.val_classes,
                                             transform=self.transform)

        elif stage == "validate":
            # Check if val_classes is provided
            self._verify_valid_classes_input_parameter(self.val_classes)

            # Create dataset for validation
            self.dataset_val = datasets.LSUN(root=self.data_dir,
                                             classes=self.val_classes,
                                             transform=self.transform)

        elif stage in ["test", "predict"]:
            # Check if test_classes is provided
            self._verify_valid_classes_input_parameter(self.test_classes)

            # Create dataset for testing
            self.dataset_test = datasets.LSUN(root=self.data_dir,
                                              classes=self.test_classes,
                                              transform=self.transform)

        if self.verbose:
            print(f"Data prepared for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for training.
        """
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for validation.
        """
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for testing.
        """
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for prediction.
        """
        return self.test_dataloader()

    def _verify_valid_classes_input_parameter(self,
                                              classes: ClassesInputType) -> None:
        """
        Verifies the classes input parameter.

        Parameters:
        ----------
            classes: str or list
                The classes to verify.

        Returns:
        ----------
            bool: True if the classes are valid, False otherwise.
        """
        if classes == [] or classes is None:
            raise ValueError("Did not provide adequate classes for the stage.")
