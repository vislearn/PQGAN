import typing
from abc import ABC, abstractmethod

import lightning as L
import torch
from torchvision import transforms


class CoreDataModule(L.LightningDataModule, ABC):
    @abstractmethod
    def get_normalization(self) -> transforms.Normalize:
        """
        Should return torch transform.Normalize object with mean and std.
        The returned object is used to normalize the data in the given dataclass.

        ----------
        Returns:
            transforms.Normalize: A transform object that normalizes the data.
        """
        pass

    def get_renormalizer(self,
                         target_min: float = 0.0,
                         target_max: float = 1.0) -> typing.Callable:
        """
        Returns a function that denormalizes a tensor using the normalization
        defined in get_normalization() and rescales the result to [target_min, target_max].

        Parameters:
        ----------
        target_min: float
            The minimum value of the rescaled tensor.
        target_max: float
            The maximum value of the rescaled tensor.

        Returns:
        ----------
        denormalize: function
            A function that takes a tensor and returns the denormalized and rescaled tensor.
        """
        # Old normalization values
        norm = self.get_normalization()
        mean_old = torch.tensor(norm.mean).view(1, -1, 1, 1)  # shape: [1, C, 1, 1]
        std_old = torch.tensor(norm.std).view(1, -1, 1, 1)    # shape: [1, C, 1, 1]

        # Assuming the original range of the normalized tensor was approximately [0, 1]
        # This is always the case for ToTensor()transform which is used in all datamodules
        original_min = torch.tensor(0.0)
        original_max = torch.tensor(1.0)
        target_min_tensor = torch.tensor(target_min)
        target_max_tensor = torch.tensor(target_max)
        range_original = original_max - original_min
        range_target = target_max_tensor - target_min_tensor

        def renormalize(tensor: torch.Tensor) -> torch.Tensor:
            # Ensure the tensors for calculations are on the same device and have compatible dtypes
            device = tensor.device
            mean_old_dev = mean_old.to(device)
            std_old_dev = std_old.to(device)
            original_min_dev = original_min.to(device)
            range_original_dev = range_original.to(device)
            range_target_dev = range_target.to(device)
            target_min_dev = target_min_tensor.to(device)

            # Step 1: Undo normalization
            denorm_tensor = tensor * std_old_dev + mean_old_dev

            # Step 2: Rescale to [target_min, target_max]
            # Use the rescaling formula directly
            rescaled_tensor = target_min_dev + (denorm_tensor - original_min_dev) * range_target_dev / range_original_dev

            return rescaled_tensor

        return renormalize
