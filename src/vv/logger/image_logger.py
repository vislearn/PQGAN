import lightning as L
import torch
import torchvision.utils as vutils
from pytorch_lightning.utilities import rank_zero_only


class ValidationImageLogger(L.Callback):
    def __init__(self,
                 num_images: int = 5,
                 tag_name: str = "images_in_vs_out"):
        """
        Logs the original images and the reconstructed images to tensorboard.
        Only works with tensorboard logger.
        Logs first `num_images` images from the last batch of the validation set.

        Parameters
        ----------
        num_images : int
            The number of images to log. If the batch size is smaller than this number, all images will be logged.
        """
        super().__init__()
        self.num_images = num_images
        self.tag_name = tag_name

    @rank_zero_only
    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule) -> None:
        """
        Checks if the logger is TensorBoard at the beginning of training.
        """
        if not isinstance(pl_module.logger, L.pytorch.loggers.TensorBoardLogger):
            raise ValueError("ValidationImageLogger requires a TensorBoardLogger.")

    @rank_zero_only
    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: dict,
                                batch: tuple[torch.Tensor, int],
                                batch_idx: int) -> None:
        """
        Logs the original images and the reconstructed images to tensorboard
        for the first batch of the validation set.

        Parameters
        ----------
        trainer : pl.Trainer
            The pytorch lightning trainer object.
        pl_module : pl.LightningModule
            The pytorch lightning module.
        outputs : dict
            The outputs of the model.
        batch : tuple[torch.Tensor, int]
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        if batch_idx == 0 and not trainer.sanity_checking:
            # Extract the relevant data for model
            x = pl_module.get_input(batch)

            if x.shape[0] >= self.num_images:
                input_images = x[:self.num_images]
                img_num = self.num_images
            else:
                input_images = x
                img_num = x.shape[0]

            # TODO: The [0] is requiered for the taming transformers model...
            output_images = pl_module.forward(input_images)[0]

            combined_images = torch.cat((input_images, output_images), dim=0)
            grid = vutils.make_grid(combined_images,
                                    nrow=img_num,
                                    normalize=True,
                                    scale_each=True)
            step = pl_module.global_step / trainer.world_size

            pl_module.logger.experiment.add_image(tag=self.tag_name,
                                                  img_tensor=grid,
                                                  global_step=step)
