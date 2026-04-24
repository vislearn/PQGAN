import os
import typing as Typing

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from vv.data.core.data_class import CoreDataModule


class ImageNetLoader(CoreDataModule):
    """
    DataModule for the ImageNet dataset.
    https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

    Attributes:
    ----------
    train_instances
        Number of instances to use for training.
    val_instances
        Number of instances to use for validation.
    test_instances
        Number of instances to use for testing.
    batch_size: int
        Batch size for the DataLoader.
    workers: int
        Number of workers for the DataLoader.
    data_dir: str
        Directory where the data is stored.
    """
    def __init__(self,
                 resolution: int = 256,
                 train_instances: int = 1281167,
                 val_instances: int = 50000,
                 test_instances: int = 100000,
                 norm_values: Typing.Optional[Typing.Tuple[float, float]] = None,
                 batch_size: int = 128,
                 workers: int = 10,
                 verbose: bool = False,
                 data_dir: str = "/path/to/imageNet") -> None:
        """
        Parameters:
        ----------
        resolution: int
            Resolution of the images.
        train_instances: int
            Number of instances to use for training.
        val_instances: int
            Number of instances to use for validation.
        test_instances: int
            Number of instances to use for testing.
        norm_values: Typing.Optional[Typing.Tuple[float, float]]
            Values for the data normalization.
            First value is the mean, second value is the std.
            If None, the default values for ImageNet are used.
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
        self.train_instances = train_instances
        self.val_instances = val_instances
        self.test_instances = test_instances
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.verbose = verbose

        if norm_values is None:
            self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
        else:
            self.normalization = transforms.Normalize(mean=[norm_values[0], norm_values[0], norm_values[0]],
                                                      std=[norm_values[1], norm_values[1], norm_values[1]])

        # Normalizing the data for spcifically the ImageNet dataset, mean and std are properties of the data
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

        for split in ["train", "val", "test"]:
            split_path = os.path.join(self.data_dir, split)
            assert os.path.exists(split_path), f"Missing {split} directory in {self.data_dir}."

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
        def load_images(root: str) -> datasets.DatasetFolder:
            """
            Load images from a directory

            Parameters:
            ----------
            root: str
                The directory to load the images from.

            Returns:
            ----------
            datasets.DatasetFolder: DatasetFolder
                The dataset containing the images
            """
            return datasets.DatasetFolder(root,
                                          loader=default_loader,
                                          extensions=(".jpg", ".jpeg", ".png"),
                                          transform=self.transform)

        if stage == "fit":
            full_dataset_train = load_images(os.path.join(self.data_dir, "train"))
            self.dataset_train = Subset(full_dataset_train, range(min(self.train_instances, len(full_dataset_train))))

            full_dataset_val = load_images(os.path.join(self.data_dir, "val"))
            self.dataset_val = Subset(full_dataset_val, range(min(self.val_instances, len(full_dataset_val))))

        if stage == "validate":
            full_dataset_val = load_images(os.path.join(self.data_dir, "val"))
            self.dataset_val = Subset(full_dataset_val, range(min(self.val_instances, len(full_dataset_val))))

        if stage == "test" or stage == "predict":
            full_dataset_test = load_images(os.path.join(self.data_dir, "test"))
            self.dataset_test = Subset(full_dataset_test, range(min(self.test_instances, len(full_dataset_test))))

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
