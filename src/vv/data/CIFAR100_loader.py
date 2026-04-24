import typing as Typing

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

import vv.utilities as utils
from vv.data.core.data_class import CoreDataModule


class CIFAR100Loader(CoreDataModule):
    """
    DataModule for the CIFAR-100 dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes:
    ----------
    data_dir: str
        Directory where the data is stored.
    train_val_split: float
        Proportion of the data to use for training.
    batch_size: int
        Batch size for the DataLoader.
    workers: int
        Number of workers for the DataLoader.
    transform: torchvision.transforms.Compose
        Transform to apply to the data.
    """
    def __init__(self,
                 train_val_split: float = 0.8,
                 norm_values: Typing.Optional[Typing.Tuple[float, float]] = None,
                 batch_size: int = 1024,
                 workers: int = 10,
                 verbose: bool = False,
                 data_dir: str = utils.get_dir("data")) -> None:
        """
        Parameters:
        ----------
        train_val_split: float
            Proportion of the data to use for training.
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
        data_dir: str
            Directory where the data is stored.
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose

        if norm_values is None:
            self.normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])
        else:
            self.normalization = transforms.Normalize(mean=[norm_values[0], norm_values[0], norm_values[0]],
                                                      std=[norm_values[1], norm_values[1], norm_values[1]])

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             self.normalization])

        self.data_size = 32 * 32 * 3  # Mnaualy set the size of the data

        self.save_hyperparameters()

    def get_normalization(self) -> transforms.Normalize:
        """
        Returns the normalization transform for the data.

        Parameters:
        ----------
        None

        Returns:
        ----------
        transforms.Normalize: A transform object that normalizes the data.
        """
        return self.normalization

    def prepare_data(self) -> None:
        """
        Downloads the dataset.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
        """
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

        if self.verbose:
            print("All downloads completed for CIFAR100 dataset.")

    def setup(self, stage: str) -> None:
        """
        Augment the data for the training, validation and test

        Parameters:
        ----------
        stage: str
            The stage of the data preparation process: [fit, validate, test, predict]

        Returns:
        ----------
        None
        """
        if stage == "fit":
            dataset_full = CIFAR100(self.data_dir, train=True, transform=self.transform)
            full_size = len(dataset_full)
            train_size = int(full_size * self.train_val_split)
            val_size = val_size = full_size - train_size
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [train_size, val_size]
            )

        if stage == "test":
            self.dataset_test = CIFAR100(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.dataset_predict = CIFAR100(self.data_dir, train=False, transform=self.transform)

        if self.verbose:
            print(f"Data prepared for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader. Data is shuffled

        Parameters:
        ----------
        None

        Returns:
        ----------
        DataLoader: DataLoader
            The training DataLoader
        """
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.

        Parameters:
        ----------
        None

        Returns:
        ----------
        DataLoader: DataLoader
            The validation DataLoader"""
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test DataLoader.

        Parameters:
        ----------
        None

        Returns:
        ----------
        DataLoader: DataLoader
            The test DataLoader"""
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction DataLoader.

        Parameters:
        ----------
        None

        Returns:
        ----------
        DataLoader: DataLoader
            The prediction DataLoader"""
        return DataLoader(self.dataset_predict,
                          batch_size=self.batch_size,
                          num_workers=self.workers)
