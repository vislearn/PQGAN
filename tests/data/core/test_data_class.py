import os

import pytest
import torch
from torchvision import transforms

from vv.data.core import CoreDataModule

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No cuda on Github.")


class TmpDataModule(CoreDataModule):
    def get_normalization(self) -> transforms.Normalize:
        # Generate random mean and std for testing
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        return transforms.Normalize(mean=mean, std=std)


@pytest.fixture
def data_module() -> CoreDataModule:
    return TmpDataModule()


@pytest.fixture
def sample_tensor_edgecase_min() -> torch.Tensor:
    return torch.zeros(1, 3, 1, 1)  # [0, 0]


@pytest.fixture
def sample_tensor_edgecase_max() -> torch.Tensor:
    return torch.ones(1, 3, 1, 1)  # [1, 1]


@pytest.mark.parametrize("renormalizer_min, renormalizer_max", [
    (-1.0, 1.0,),
    (-2.0, 2.0,),
    (0.0, 1.0),
    (-0.5, 0.5,),
])
def test_edge_cases(data_module: CoreDataModule,
                    sample_tensor_edgecase_min: torch.Tensor,
                    sample_tensor_edgecase_max: torch.Tensor,
                    renormalizer_min: float,
                    renormalizer_max: float) -> None:
    """
    Test the denormalizer with edge cases.
    """
    # Rename
    min = sample_tensor_edgecase_min
    max = sample_tensor_edgecase_max

    # Normalize data
    norm_old = data_module.get_normalization()
    min_old = norm_old(min)
    max_old = norm_old(max)

    # Check if first normalization is correct
    expected_min_old = -2.5
    expected_max_old = 2.5
    expected_min_old_tensor = torch.tensor([expected_min_old, expected_min_old, expected_min_old]).view(1, 3, 1, 1)
    expected_max_old_tensor = torch.tensor([expected_max_old, expected_max_old, expected_max_old]).view(1, 3, 1, 1)

    assert torch.equal(min_old, expected_min_old_tensor)
    assert torch.equal(max_old, expected_max_old_tensor)
    assert min_old.shape == torch.Size([1, 3, 1, 1])
    assert max_old.shape == torch.Size([1, 3, 1, 1])

    # Get new normalizer
    renormalizer = data_module.get_renormalizer(renormalizer_min, renormalizer_max)

    # Renormalize data
    min_new = renormalizer(min_old)
    max_new = renormalizer(max_old)

    # Check if second normalization is correct
    expected_min_new = renormalizer_min
    expected_max_new = renormalizer_max
    expected_min_new_tensor = torch.tensor([expected_min_new, expected_min_new, expected_min_new]).view(1, 3, 1, 1)
    expected_max_new_tensor = torch.tensor([expected_max_new, expected_max_new, expected_max_new]).view(1, 3, 1, 1)
    assert torch.equal(min_new, expected_min_new_tensor)
    assert torch.equal(max_new, expected_max_new_tensor)
    assert min_new.shape == torch.Size([1, 3, 1, 1])
    assert max_new.shape == torch.Size([1, 3, 1, 1])


@pytest.mark.parametrize("renormalizer_min, renormalizer_max, sample_tensor, expected_tensor", [
    (-1.0,
     1.0,
     torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
     torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)),
    (-0.7,
     0.5,
     torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
     torch.tensor([-0.1, -0.1, -0.1]).view(1, 3, 1, 1)),
    (-0.7,
     0.5,
     torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1),
     torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)),
    (0.0,
     1.0,
     torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1),
     torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)),
    (-1.0,
     1.0,
     torch.tensor([0.2, 0.2, 0.2]).view(1, 3, 1, 1),
     torch.tensor([-0.6, -0.6, -0.6]).view(1, 3, 1, 1)),
    (-0.7,
     0.5,
     torch.tensor([0.9, 0.9, 0.9]).view(1, 3, 1, 1),
     torch.tensor([0.38, 0.38, 0.38]).view(1, 3, 1, 1)),
])
def test_normal_cases(data_module: CoreDataModule,
                      sample_tensor: torch.Tensor,
                      expected_tensor: torch.Tensor,
                      renormalizer_min: float,
                      renormalizer_max: float) -> None:
    """
    Test the denormalizer with normal cases.
    """
    # Normalize data
    norm_old = data_module.get_normalization()
    sample_old = norm_old(sample_tensor)

    # Get new normalizer
    renormalizer = data_module.get_renormalizer(renormalizer_min, renormalizer_max)

    # Renormalize data
    sample_new = renormalizer(sample_old)

    assert sample_new.shape == torch.Size([1, 3, 1, 1])
    assert torch.allclose(sample_new, expected_tensor, atol=1e-6)
