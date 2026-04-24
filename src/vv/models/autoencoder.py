import lightning as L
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchviz import make_dot

from vv.models.core import BaseModel


class Autoencoder(L.LightningModule):
    """
    Autoencoder model that is assembled from an encoder and a decoder.

    Attributes:
    ----------
    encoder : BaseModel
        The encoder model.
    decoder : BaseModel
        The decoder model.
    latent_error_weight : float
        The weight of the latent error.
    learning_rate : float
        The learning rate for the optimizer.
    lr_patience : int
        The patience for the learning rate scheduler.
    disconnect_decoder_loss_propagation_from_encoder:
        Whether to disconnect the decoder loss propagation from the encoder. The loss of the decoder will not be
        propagated to the VQ if this is set to True.
    """
    def __init__(self,
                 encoder: BaseModel,
                 decoder: BaseModel,
                 disconnect_decoder_loss_propagation_from_encoder: bool = False,
                 latent_error_weight: float = 1.0,
                 learning_rate: float = 1e-3,
                 lr_patience: int = 1) -> None:
        """
        Build autoencoder from encoder and decoder. Hyperparameters are for training.

        Parameters:
        -----------
        encoder : BaseModel
            The encoder model.
        decoder : BaseModel
            The decoder model.
        disconnect_decoder_loss_propagation_from_encoder:
            Whether to disconnect the decoder loss propagation from the encoder. The loss of the decoder will not be
            propagated to the VQ if this is set to True.
        latent_error_weight : float
            The weight of the latent error.
        learning_rate : float
            The learning rate for the optimizer.
        lr_patience : int
            The patience for the learning rate scheduler.

        Returns:
        --------
        None
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.disconnect_decoder_loss_propagation_from_encoder = disconnect_decoder_loss_propagation_from_encoder
        self.latent_error_weight = latent_error_weight
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience

        self.example_input_array = torch.Tensor(1,
                                                encoder.input_size[0],
                                                encoder.input_size[1],
                                                encoder.input_size[2])
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The encoded tensor.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the input tensor.

        Parameters:
        -----------
        z : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The decoded tensor. Flattend
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Parameters:
        -----------
        x: torch.Tensor
            The input data x

        Returns:
        --------
        torch.Tensor
            The output tensor. Same shape as the input tensor.
        """
        if self.disconnect_decoder_loss_propagation_from_encoder:
            z = self.encode(x).detach()
            x_hat = self.decode(z)
        else:
            z = self.encode(x)
            x_hat = self.decode(z)

        return x_hat

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the autoencoder if it is a generative model.

        Parameters:
        -----------
        n_samples : int
            The number of samples to generate.

        Returns:
        --------
        torch.Tensor
            The generated samples.
        """
        if hasattr(self.encoder, 'sample'):
            z = self.encoder.sample(n_samples)
            x_hat = self.decode(z)
            return x_hat
        else:
            raise AttributeError("Encoder does not have a sample method.")

    def training_step(self, batch: tuple[torch.Tensor, int]) -> torch.Tensor:
        """
        Training loop for the autoencoder.

        Parameters:
        -----------
        batch: tuple[torch.Tensor, int]
            The input data batch x and the corresponding labels y.

        Returns:
        --------
        loss : torch.Tensor
            The loss of the autoencoder.
        """
        x, y = batch

        x_hat = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x)
        encoder_loss = self.latent_error_weight * self.encoder.get_additional_loss_objective()
        total_loss = reconstruction_loss + encoder_loss

        # Log the gradients of the trainable parameters
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(f'gradients/{name}', param.grad.norm(), on_step=True, on_epoch=True, sync_dist=True)

        self.log("train_encoder_loss", encoder_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    @rank_zero_only
    def __save_images(self, x: torch.Tensor) -> None:
        """
        Save images of the fixed images.

        Parameters:
        -----------
        x: torch.Tensor
            Validation batch.

        Returns:
        --------
        None
        """
        NUM_IMAGES = 5

        # Select data for visualization.
        # If batch is smaller than NUM_IMAGES, use the whole batch.
        if x.shape[0] >= NUM_IMAGES:
            fixed_images = x[:NUM_IMAGES]
            img_num = NUM_IMAGES
        else:
            fixed_images = x
            img_num = x.shape[0]

        fixed_preds = self.forward(fixed_images)

        combined_images = torch.cat((fixed_images, fixed_preds), dim=0)
        grid = vutils.make_grid(combined_images,
                                nrow=img_num,
                                normalize=True,
                                scale_each=True)
        self.logger.experiment.add_image(tag="original_vs_reconstruction",
                                         img_tensor=grid,
                                         global_step=self.global_step)

    def validation_step(self, batch: tuple[torch.Tensor, int], batch_idx: int) -> None:
        """
        Validation loop for the autoencoder.

        Parameters:
        -----------
        batch: tuple[torch.Tensor, int]
            The input data batch x and the corresponding labels y.
        batch_idx: int
            The index of the batch.

        Returns:
        --------
        loss : torch.Tensor
            The loss of the autoencoder.
        """
        x, _ = batch

        # Run validation
        x_hat = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x)
        encoder_loss = self.latent_error_weight * self.encoder.get_additional_loss_objective()
        total_loss = reconstruction_loss + encoder_loss

        self.log("val_encoder_loss", encoder_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_reconstruction_loss", reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Save images
        if batch_idx == 0:
            self.__save_images(x)

    def test_step(self, batch: tuple[torch.Tensor, int]) -> torch.Tensor:
        """
        Test loop for the autoencoder.

        Parameters:
        -----------
        batch: tuple[torch.Tensor, int]
            The input data batch x and the corresponding labels y.

        Returns:
        --------
        loss : torch.Tensor
            The loss of the autoencoder.
        """
        x, y = batch

        x_hat = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x)
        encoder_loss = self.latent_error_weight * self.encoder.get_additional_loss_objective()
        total_loss = reconstruction_loss + encoder_loss

        self.log("test_encoder_loss", encoder_loss, sync_dist=True)
        self.log("test_reconstruction_loss", reconstruction_loss, sync_dist=True)
        self.log("test_loss", total_loss, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for the autoencoder.

        Parameters:
        -----------
        None

        Returns:
        --------
        optimizer : torch.optim.Optimizer
            The optimizer for the autoencoder.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer=optimizer,
                                               mode="min",
                                               factor=0.1,
                                               patience=self.lr_patience,
                                               threshold=1e-4,
                                               threshold_mode="rel"),
                "monitor": "val_loss"
            },
        }

    def visualize_gradient_flow(self, x: torch.Tensor,
                                filepath: str = "gradient_flow",
                                show_gradients: bool = True) -> None:
        """
        Visualize the gradient flow through the network using torchviz.

        Parameters:
        -----------
        x: torch.Tensor
            The input tensor.
        filepath: str
            The filepath to save the visualization.

        Returns:
        --------
        None
        """
        # Save the original state of automatic_optimization
        original_auto_optimization: bool = self.automatic_optimization  # type: ignore
        self.automatic_optimization = False

        try:
            x.requires_grad_(True)  # Ensure input tensor requires gradient
            x_hat = self.forward(x)

            reconstruction_loss = F.mse_loss(x_hat, x)
            encoder_loss = self.latent_error_weight * self.encoder.get_additional_loss_objective()
            total_loss = reconstruction_loss + encoder_loss

            self.backward(total_loss,
                          retain_graph=True)
            dot = make_dot(var=total_loss,
                           params=dict(self.named_parameters()),
                           show_attrs=show_gradients,
                           show_saved=show_gradients)
            dot.format = 'png'
            dot.render(filepath)
        finally:
            self.automatic_optimization = original_auto_optimization
