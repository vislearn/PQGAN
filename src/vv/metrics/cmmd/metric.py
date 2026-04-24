import torch
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn

from vv.metrics.cmmd import distance, embedding

_RECOMMENDED_SAMPLE_NUMBER = 5000


class CMMDMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

        rank_zero_warn(
            "Metric `Clip MMD` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        self.embedding_model = embedding.ClipEmbeddingModel()

        self.add_state("all_preds_embed", default=[], dist_reduce_fx="cat")  # type: ignore
        self.add_state("all_target_embed", default=[], dist_reduce_fx="cat")  # type: ignore

    def _sanityCheck(self,
                     preds: torch.Tensor,
                     target: torch.Tensor) -> None:
        """
        Check if the input tensors are valid.
        Args:
            preds: Predictions tensor.
            target: Target tensor.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if torch.min(preds) < 0 or torch.max(preds) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{torch.min(preds)}, {torch.max(preds)}]."
            )
        if torch.min(target) < 0 or torch.max(target) > 1:
            raise ValueError(
                "Image values are expected to be in [0, 1]. Found:" f" [{torch.min(preds)}, {torch.max(preds)}]."
            )

    def update(self,
               preds: torch.Tensor,
               target: torch.Tensor) -> None:
        """
        Update the state with the predictions and target tensors.
        Args:
            preds: Predictions tensor. Shape should be (batch_size, 3, height, width) and values in [0, 1].
            target: Target tensor. Shape should be (batch_size, 3, height, width) and values in [0, 1].
        """
        self._sanityCheck(preds, target)

        preds_embed = self.embedding_model.embed(preds)
        target_embed = self.embedding_model.embed(target)

        self.all_preds_embed.append(preds_embed)  # type: ignore
        self.all_target_embed.append(target_embed)  # type: ignore

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric by concatenating all embeddings and calculating MMD.

        Returns:
            torch.Tensor: The computed MMD distance.
        """
        sample_number = len(self.all_preds_embed)  # type: ignore

        if sample_number == 0:
            raise RuntimeError("Cannot compute MMD: no embeddings were provided.")
        if sample_number < _RECOMMENDED_SAMPLE_NUMBER:
            rank_zero_warn(f"MMD computation might not be reliable with {sample_number} samples.",
                           UserWarning)

        if type(self.all_preds_embed) is list:  # type: ignore
            rank_zero_warn(
                "The state is a list, indicating that the tensors have not been reduced yet. "
                "Concatenating all tensors to compute the final MMD"
                "In a distributed setting, this may lead to incorrect reduction."
                "Please ensure the strategy calls the sync() function of the metric.",
                UserWarning,
            )
            self.all_preds_embed = torch.cat(self.all_preds_embed, dim=0)  # type: ignore
            self.all_target_embed = torch.cat(self.all_target_embed, dim=0)  # type: ignore
        else:
            # If the state is already a tensor, we assume it has been reduced
            # and we don't need to concatenate again.
            pass

        # torch.cat is called as defined in the dist_reduce_fx and converts from list to tensor
        return distance.mmd(self.all_preds_embed, self.all_target_embed)
