from dataclasses import dataclass
from typing import Union


@dataclass
class FIDConfig:
    """
    Configuration for the Frechet Inception Distance (FID) metric.
    Lower is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    feature_layer_size : int
        The size of the feature layer used for FID calculation.
    """
    active: bool = True
    feature_layer_size: int = 2048


@dataclass
class KIDConfig:
    """
    Configuration for the Kernel Inception Distance (KID) metric.
    Lower is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    feature_layer_size : int
        The size of the feature layer used for KID calculation.
    """
    active: bool = True
    feature_layer_size: int = 2048


@dataclass
class ISConfig:
    """
    Configuration for the Inception Score (IS) metric.
    Higher is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    feature_layer_size : int
        The size of the feature layer used for IS calculation.
    splits : int
        The number of splits for IS calculation.
    """
    active: bool = True
    feature_layer_size: int = 2048
    splits: int = 10


@dataclass
class LPIPSConfig:
    """
    Configuration for the Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Lower is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    net_type : str
        The model used for LPIPS calculation.
    reduction : str
        The reduction method used for LPIPS calculation.
    """
    active: bool = True
    net_type: str = "alex"  # Options: "alex", "vgg", "squeeze"
    reduction: str = "mean"  # Options: "mean", "sum"


@dataclass
class PSNRConfig:
    """
    Configuration for the Peak Signal-to-Noise Ratio (PSNR) metric.
    Higher is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    data_range : Union[float, tuple[float, float], None]
        The range of values in the input data.
    base : int
        The base used for logarithmic calculations.
    reduction : str
        The reduction method used for PSNR calculation.
    dim : Union[int, tuple[int, ...], None]
        The dimensions to reduce over.
    """
    active: bool = True
    data_range: Union[float, tuple[float, float], None] = (0.0, 1.0)
    base: int = 10
    reduction: str = "elementwise_mean"  # Options: "elementwise_mean", "sum", "none"
    dim: Union[int, tuple[int, ...], None] = None


@dataclass
class CMMDConfig:
    """
    Configuration for the Clip MMD metric.
    Lower is better.

    Attributes
    ----------
    active : bool
        Whether to use the FID metric.
    """
    active: bool = True
