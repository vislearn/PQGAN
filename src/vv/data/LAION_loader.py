import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from vv.data.core.data_class import CoreDataModule


class LAIONLoader(CoreDataModule):
    """
    DataModule for the LAION dataset.
    """
    def __init__(self,
                 resolution: int = 256,
                 val_split: str = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/LAION-AE/LAION_10k_split.txt",
                 norm_values: Optional[Tuple[float, float]] = None,
                 batch_size: int = 32,
                 workers: int = 20,
                 verbose: bool = False,
                 data_dir: str = "/export/data/vislearn/rother_subgroup/dzavadsk/datasets/LAION-AE/images/0to500k/") -> None:
        """
        Parameters:
        ----------
        resolution: int
            Target image resolution.
        val_split: str
            Fraction of the data to use for validation. Path to the validation split file.
        norm_values: Tuple[float, float]
            Normalization (mean, std). E.g., (0.5, 0.5) for [-1, 1] range.
        batch_size: int
            Batch size for the DataLoaders.
        workers: int
            Number of DataLoader workers.
        verbose: bool
            Print setup info if True.
        data_dir: str
            Path to dataset root directory.
        """
        super().__init__()
        self.resolution = resolution
        self.val_split = val_split

        self.batch_size = batch_size
        self.workers = workers
        self.data_dir = data_dir
        self.verbose = verbose

        if norm_values is None:
            self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
        else:
            self.normalization = transforms.Normalize(mean=[norm_values[0]] * 3,
                                                      std=[norm_values[1]] * 3)

        self.transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),
            transforms.ToTensor(),
            self.normalization
        ])

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
        assert os.path.isfile(self.val_split), f"Validation split file not found: {self.val_split}"

    def _load_from_txt(self,
                       file_path: str) -> Dataset:
        """
        Loads dataset from a list of image paths in a txt file.

        Parameters:
        ----------
        txt_path: str
            Path to the txt file containing image paths.

        Returns:
        ----------
        Dataset: Dataset
            A PyTorch Dataset object containing the images.
        """
        with open(file_path, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        if self.verbose:
            print(f"Loaded {len(paths)} images from {file_path}")

        class TxtDataset(Dataset):
            def __init__(self,
                         paths: list[str],
                         transform: transforms.Compose = self.transform) -> None:
                self.paths = paths
                self.transform = transform

            def __len__(self) -> int:
                return len(self.paths)

            def __getitem__(self,
                            idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                path = self.paths[idx]
                image = default_loader(path)
                dummy_label = torch.zeros(1)
                return self.transform(image), dummy_label

        return TxtDataset(paths, self.transform)

    def setup(self, stage: str) -> None:
        if stage == "validate":
            self.dataset_val = self._load_from_txt(self.val_split)
        else:
            raise NotImplementedError(f"Stage '{stage}' is not supported in this barebones setup.")

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
