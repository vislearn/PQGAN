import os
import typing as Typing

from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from vv.data.core.data_class import CoreDataModule


class FFHQLoader(CoreDataModule):
    """
    DataModule for the FFHQ dataset. There are 70,000 images in the dataset.
    https://www.kaggle.com/datasets/gibi13/flickr-faces-hq-dataset-ffhq
    """
    def __init__(self,
                 resolution: int = 1024,
                 test_split: float = 0.14,
                 train_val_split: float = 0.8,
                 norm_values: Typing.Optional[Typing.Tuple[float, float]] = None,
                 batch_size: int = 32,
                 workers: int = 20,
                 verbose: bool = False,
                 data_dir: str = "/path/to/ffhq") -> None:
        """
        Parameters:
        ----------
        resolution: int
            Resolution of the images.
        test_split: float
            Fraction of the data to use for testing.
            14% = 10,000 images
        train_val_split: float
            Fraction of the data to use for training.
            80% = 48,000 training images and 12,000 validation images
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
        self.test_split = test_split
        self.train_val_split = train_val_split

        self.batch_size = batch_size
        self.workers = workers
        self.data_dir = data_dir
        self.verbose = verbose

        if norm_values is None:
            self.normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])
        else:
            self.normalization = transforms.Normalize(mean=[norm_values[0], norm_values[0], norm_values[0]],
                                                      std=[norm_values[1], norm_values[1], norm_values[1]])

        self.transform = transforms.Compose([transforms.Resize((self.resolution, self.resolution)),
                                             transforms.ToTensor(),
                                             self.normalization])

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
        Verifies the data is downloaded and stored in the correct directory.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None
        """
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."

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
        # Get full dataset size once
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform, loader=default_loader)
        dataset_size = len(full_dataset)

        # Define fixed test split
        test_size = int(self.test_split * dataset_size)
        test_indices = list(range(test_size))
        train_val_indices = list(range(test_size, dataset_size))

        if stage == "fit":
            train_val_dataset = Subset(full_dataset, train_val_indices)
            train_size = int(self.train_val_split * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size

            self.dataset_train, self.dataset_val = random_split(train_val_dataset, [train_size, val_size])

        elif stage == "validate":
            train_val_dataset = Subset(full_dataset, train_val_indices)
            train_size = int(self.train_val_split * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size

            _, self.dataset_val = random_split(train_val_dataset, [train_size, val_size])

        elif stage in ["test", "predict"]:
            self.dataset_test = Subset(full_dataset, test_indices)

        if self.verbose:
            print(f"Dataset prepared for stage: {stage}")
            if stage == "fit":
                print(f"Train size: {len(self.dataset_train)}")
                print(f"Validation size: {len(self.dataset_val)}")
            elif stage in ["test", "predict"]:
                print(f"Test size: {len(self.dataset_test)}")

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
            The validation DataLoader
        """
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
            The test DataLoader
        """
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
            The prediction DataLoader
        """
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers)
