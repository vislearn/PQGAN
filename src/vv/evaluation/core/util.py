from typing import Any, Dict, List, Tuple

import torch

from vv.shared.shared_types import CodebookUsage


def _concatenate_codebooks(codebook_statistics: CodebookUsage) -> torch.Tensor:
    """
    Concatenates list of bincount tensors into one tensor.

    Parameters
    ----------
    codebook_statistics : CodebookUsage
        List of CodebookUsageEntry objects containing bincount tensors.

    Returns
    ----------
    torch.Tensor
        Concatenated tensor of usage counts.
    """
    usage_counts = torch.cat([b.bincount.cpu() for b in codebook_statistics], dim=0)
    return usage_counts


def _compute_entropy(codebook_statistics: torch.Tensor) -> Tuple[float, float]:
    """
    Computes entropy and normalized entropy from usage counts.

    Parameters
    ----------
    codebook_statistics : torch.Tensor
        Tensor of usage counts.

    Returns
    ----------
    tuple
        - entropy: float
            The computed entropy.
        - normalized_entropy: float
            The normalized entropy.
    """
    EPSILON = 1e-12  # Small value to avoid division by zero

    total = codebook_statistics.sum()
    probs = codebook_statistics.float() / (total + EPSILON)

    mask = probs > 0
    entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))

    K = codebook_statistics.shape[0]
    max_entropy = torch.log(torch.tensor(K, dtype=probs.dtype))
    normalized_entropy = entropy / (max_entropy + EPSILON)

    return entropy.item(), normalized_entropy.item()


def _compute_perplexity(usage_counts: torch.Tensor) -> Tuple[float, float]:
    """
    Computes perplexity and normalized perplexity from usage counts.

    Parameters
    ----------
    usage_counts : torch.Tensor
        Tensor of usage counts.

    Returns
    ----------
    tuple
        - perplexity: float
            The computed perplexity.
        - normalized_perplexity: float
            The normalized perplexity.
    """
    entropy, _ = _compute_entropy(usage_counts)
    perplexity = torch.exp(torch.tensor(entropy))

    K = usage_counts.shape[0]
    normalized_perplexity = perplexity / K

    return perplexity.item(), normalized_perplexity.item()


def _compute_codebook_utilization(codebook_statistics: CodebookUsage) -> List[float]:
    """
    Compute per-sub-codebook utilization rate (binary usage %).

    Parameters
    ----------
    codebook_statistics : CodebookUsage
        List of CodebookUsageEntry objects containing bincount tensors.

    Returns
    ----------
    list
        List of utilization rates for each sub-codebook.
    """
    utilization_rates = []
    for statistic in codebook_statistics:
        tensor = statistic.bincount.cpu()
        utilization_rate = (tensor != 0).float().mean().item()
        utilization_rates.append(utilization_rate)
    return utilization_rates


def _compute_avg_codebook_utilization(codebook_statistics: CodebookUsage) -> float:
    """
    Compute global codebook utilization rate across all entries.

    Parameters
    ----------
    codebook_statistics : CodebookUsage
        List of CodebookUsageEntry objects containing bincount tensors.

    Returns
    ----------
    float
        Average utilization rate across all sub-codebooks.
    """
    total_entries = 0
    non_zero_entries = 0

    for statistic in codebook_statistics:
        tensor = statistic.bincount.cpu()
        total_entries += tensor.numel()
        non_zero_entries += (tensor != 0).sum().item()

    utilization_rate = non_zero_entries / total_entries if total_entries > 0 else 0.0
    return utilization_rate


def codebook_statistics_summary(codebook_statistics: CodebookUsage) -> Dict[str, Any]:
    """
    Compute and summarize codebook utilization, entropy, and perplexity metrics.

    Parameters
    ----------
    codebook_statistics : CodebookUsage
        List of CodebookUsageEntry objects containing bincount tensors.

    Returns
    ----------
    dict
        Dictionary containing computed metrics:
        - entropy
        - normalized_entropy
        - perplexity
        - normalized_perplexity
        - utilization_per_subcodebook
        - avg_utilization
    """
    usage_counts = _concatenate_codebooks(codebook_statistics)

    # Entropy & Perplexity
    entropy, norm_entropy = _compute_entropy(usage_counts)
    perplexity, norm_perplexity = _compute_perplexity(usage_counts)

    # Utilization Metrics
    utilization_per_subcodebook = _compute_codebook_utilization(codebook_statistics)
    avg_utilization = _compute_avg_codebook_utilization(codebook_statistics)

    return {
        'entropy': entropy,
        'normalized_entropy': norm_entropy,
        'perplexity': perplexity,
        'normalized_perplexity': norm_perplexity,
        'utilization_per_subcodebook': utilization_per_subcodebook,
        'avg_utilization': avg_utilization
    }
