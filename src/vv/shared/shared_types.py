from dataclasses import dataclass
from typing import List

import torch


@dataclass
class CodebookUsageEntry:
    codebook_index: int
    bincount: torch.Tensor


CodebookUsage = List[CodebookUsageEntry]
