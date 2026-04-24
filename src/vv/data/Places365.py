import os
import typing as Typing

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vv.data.core.data_class import CoreDataModule


class Places365Loader(CoreDataModule):
    """
    DataModule for Places365 dataset.

    1.8 million train samples
    36,000 validation samples
    K=365 scene classes
    """
    def __init__(self,
                 resolution: int = 256,
                 norm_values: Typing.Optional[Typing.Tuple[float, float]] = None,
                 batch_size: int = 128,
                 workers: int = 10,
                 verbose: bool = False,
                 data_dir: str = "/path/to/places365") -> None:
        """
        Parameters:
        ----------
        resolution: int
            Resolution of the images.
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
        Returns the normalization transform for the data.

        Returns:
        ----------
        transforms.Normalize: A transform object that normalizes the data.
        """
        return self.normalization

    def prepare_data(self) -> None:
        """
        Prepares the data by checking if the LSUN directory exists.
        """
        assert os.path.exists(self.data_dir), f"Places365 directory not found at {self.data_dir}"

    def setup(self,
              stage: str) -> None:
        """
        Prepares the data for training, validation and testing.

        Parameters:
        ----------
        stage: str
            The stage for which the data is being prepared. Can be one of
            'fit', 'validate', 'test', or 'predict'.
        """
        if self.resolution <= 256:
            self.small = True
        else:
            self.small = False

        if stage == "fit":
            self.dataset_train = datasets.Places365(root=self.data_dir,
                                                    split="train-standard",
                                                    small=self.small,
                                                    download=False,
                                                    transform=self.transform)
            self.dataset_val = datasets.Places365(root=self.data_dir,
                                                  split="val",
                                                  small=self.small,
                                                  download=False,
                                                  transform=self.transform)

        elif stage == "validate":
            self.val_dataset = datasets.Places365(root=self.data_dir,
                                                  split="val",
                                                  small=self.small,
                                                  download=False,
                                                  transform=self.transform)

        elif stage in ["test", "predict"]:
            val_dataset = datasets.Places365(root=self.data_dir,
                                             split="val",
                                             small=self.small, download=False,
                                             transform=self.transform)
            self.dataset_test = val_dataset

        if self.verbose:
            print(f"[Places365Loader] Data prepared for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.
        """
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.
        """
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test DataLoader.
        """
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the prediction DataLoader.
        """
        return self.test_dataloader()
