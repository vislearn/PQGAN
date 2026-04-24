import lightning as L
import torch
from pytorch_lightning.utilities import rank_zero_only

from vv.shared.shared_types import CodebookUsage, CodebookUsageEntry


class CodebookAnalyzer(L.Callback):
    def __init__(self,
                 tag_name: str = "codebook_utilization") -> None:
        """
        Logs a historgram to the logger, that containst information about the codebook utilization.
        The entires of the histogram are the number of times each codebook entry was used.
        Requires a logger with an `add_histogram` method, such as TensorBoardLogger.

        Parameters
        ----------
        num_names: str
            The name of the histogram. This will be used as the tag in TensorBoard.
        """
        super().__init__()
        self.tag_name = tag_name
        self._sanity_check_passed = False

    @rank_zero_only
    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule) -> None:
        """
        Performs sanity checks on the logger and the module.
        """
        self._sanity_check(trainer, pl_module)

    def on_validation_epoch_start(self,
                                  trainer: L.Trainer,
                                  pl_module: L.LightningModule) -> None:
        """
        Performs sanity checks on the logger and the module.
        Resets the histogram tensors

        Parameters
        ----------
        trainer : pl.Trainer
            The pytorch lightning trainer object.
        pl_module : pl.LightningModule
            The pytorch lightning module.

        Returns
        ----------
        None
        """
        self._sanity_check(trainer, pl_module)

        self.codebook_statistics: CodebookUsage = [
            CodebookUsageEntry(
                codebook_index=i,
                bincount=torch.zeros(pl_module.quantize.n_e, device=pl_module.device)
            )
            for i in range(pl_module.quantize.num_spaces)
        ]

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: dict,
                                batch: torch.Tensor,
                                batch_idx: int) -> None:
        """
        Called when the validation batch ends.
        This method adds the count of the batch to the histogram tensor.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.
        outputs : dict
            The outputs of the validation step.
        batch : torch.Tensor
            The input batch for the validation.
        batch_idx : int
            The index of the current batch.

        Returns
        ----------
        None
        """
        if not trainer.sanity_checking:
            # Get the codebook usage from the outputs
            codebook_usage: CodebookUsage = outputs["codebook_usage"]

            # Add the codebook usage to the histogram tensor
            for i, usage in enumerate(codebook_usage):
                self.codebook_statistics[i].bincount += usage.bincount

    def on_validation_epoch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule) -> None:
        """
        Called when the validation epoch ends.
        This method logs the histogram to the logger.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.

        Returns
        ----------
        None
        """
        if not trainer.sanity_checking:
            # Log the histogram to the logger
            logger = self._get_logger(pl_module)
            # Do it for each space seperatly to be compatible with TensorBoard
            for statistic in self.codebook_statistics:
                codebook_num: int = statistic.codebook_index
                histogram: torch.Tensor = statistic.bincount
                logger.add_histogram(
                    tag=f"{self.tag_name}_space_{codebook_num}",
                    values=histogram,
                    global_step=pl_module.global_step / trainer.world_size
                )

    def _get_logger(self,
                    pl_module: L.LightningModule) -> L.pytorch.loggers:
        """
        Returns the logger from the pl_module.
        This is needed becasue TensorBoardLogger is slightly different from the other loggers.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The pytorch lightning module.

        Returns
        ----------
        logger : pl.loggers.base.Logger
            The logger from the trainer.
        """
        logger = pl_module.logger
        return getattr(logger, "experiment", logger)

    @rank_zero_only
    def _sanity_check(self,
                      trainer: L.Trainer,
                      pl_module: L.LightningModule) -> None:
        """
        Checks if if required attributes are present in the logger and the module.

        Parameters
        ----------
        trainer : pl.Trainer
            The pytorch lightning trainer object.
        pl_module : pl.LightningModule
            The pytorch lightning module.

        Returns
        ----------
        None
        """
        if self._sanity_check_passed:
            return

        # Perform sanity checks

        # Logger
        if not hasattr(self._get_logger(pl_module), "add_histogram"):
            raise ValueError("CodebookAnalyzer requires a logger with an 'add_histogram' method.")

        # Module
        if not hasattr(pl_module, "quantize") or not isinstance(pl_module.quantize, torch.nn.Module):
            raise ValueError(
                "The provided LightningModule must have a 'quantize' attribute of type 'torch.nn.Module'."
            )

        if not hasattr(pl_module.quantize, "embeddings"):
            raise ValueError(
                "The 'quantize' module must have an 'embeddings' attribute of type 'torch.nn.ModuleList' or 'torch.nn.Embedding'."
            )

        if isinstance(pl_module.quantize.embeddings, torch.nn.ModuleList):
            if not all(isinstance(module, torch.nn.Embedding) for module in pl_module.quantize.embeddings):
                raise ValueError(
                    "If 'quantize.embeddings' is a 'torch.nn.ModuleList', all its elements must be of type 'torch.nn.Embedding'."
                )
        elif not isinstance(pl_module.quantize.embeddings, torch.nn.Embedding):
            raise ValueError(
                "The 'quantize.embeddings' attribute must be either a 'torch.nn.ModuleList' containing 'torch.nn.Embedding' modules or a single 'torch.nn.Embedding'."
            )

        if not hasattr(pl_module, "analyze_codebook_usage") or not pl_module.analyze_codebook_usage:
            raise ValueError(
                "The provided LightningModule must have the 'analyze_codebook_usage' attribute set to True. "
                "This likely means the module does not support codebook logging or the feature is disabled."
            )

        if not hasattr(pl_module.quantize, "n_e") or not isinstance(pl_module.quantize.n_e, int):
            raise ValueError("The 'quantize' module must have an integer attribute 'n_e' describing the amount of codebook entries K.")

        if not hasattr(pl_module.quantize, "num_spaces") or not isinstance(pl_module.quantize.num_spaces, int):
            raise ValueError("The 'quantize' module must have an integer attribute 'num_spaces' describing the amount of splits S.")

        self._sanity_check_passed = True
