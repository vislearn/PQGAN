from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vv.shared.shared_types import CodebookUsage


@dataclass
class MetricResult:
    name: str
    value: float
    details: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str = field(default_factory=str)
    timestamp: str = field(default_factory=str)
    step: int = field(default_factory=int)
    metrics: List[MetricResult] = field(default_factory=list)
    codebook_statistics: Optional[CodebookUsage] = field(default_factory=list)
    metadata: Optional[Dict[str, str]] = field(default_factory=dict)

# Example usage
# result = ExperimentResult(
#     experiment_id="exp_001",
#     metrics=[
#         MetricResult(name="accuracy", value=0.95),
#         MetricResult(name="loss", value=0.05, details={"validation": 0.04, "training": 0.06}),
#     ],
#     metadata={"model": "VQ-VAE", "dataset": "CIFAR-10"}
# )
