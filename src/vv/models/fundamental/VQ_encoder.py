import typing

import torch
import torch.nn as nn
from torchinfo import summary

from vv.models.core import BaseEncoder
from vv.special_layers import VectorQuantizer


class VQEncoder(BaseEncoder):
    """
    Vector quantization (VQ) encoder model with a configurable number of layers,dropout probability and batch normalization
    ----------

    Attributes
    ----------
    additional_loss_objective: torch.Tensor
        The additional loss objective
    base_encoder : BaseEncoder
        The base encoder model
    codenook_splits: typing.Tuple[int, ...]
        Splits the respective dimensions into several heads
    codebook_size: int
        The number of discrete latent variables per dimension
    decoder_encoder_ste: bool
        Whether to use the straight through estimator from the decoder to the encoder
    use_commitment_loss: bool
        Whether to use the commitment loss for the encoder to the embedding
    vq_loss: str
        The loss algorithm to use for updating the embeddings
    embedding_loss_weight: float
        The commitment cost
    verbose : bool
        Prints the model summary
    quantizer: nn.Module
        The quantization layer
    """
    def __init__(self,
                 base_encoder: BaseEncoder,
                 codebook_splits: typing.Tuple[int, ...],
                 separate_codebooks: typing.Tuple[bool, ...],
                 codebook_size: int,
                 decoder_encoder_ste: bool,
                 use_commitment_loss: bool,
                 vq_loss: str,
                 embedding_loss_weight: float,
                 verbose: bool = False) -> None:
        """
        Variational encoder model based on a base encoder like ConvolutionalEncoder or LinearEncoder

        Parameters
        ----------
        base_encoder : BaseEncoder
            The base encoder model
        codenook_splits: torch.Size
            Splits the respective dimensions into several heads
        separate_codebooks: typing.Tuple[bool, ...]
            The number of codebooks to use. Each dimention that is split can have an adjustable number of codebooks
            The number of codebooks is 1 (False) or as many as there are splits (True)
        codebook_size: int
            The number of discrete latent variables per dimension
        decoder_encoder_ste: bool
            Whether to use the straight through estimator from the decoder to the encoder
        use_commitment_loss: bool
            Whether to use the commitment loss for the encoder to the embedding
        vq_loss: str
            The loss algorithm to use for updating the embeddings
        embedding_loss_weight: float
            The commitment cost
        verbose : bool
            Prints the model summary
        """
        super(VQEncoder, self).__init__()

        # Base encoder
        self.base_encoder = base_encoder

        # Embedding space settings
        self.codebook_size = codebook_size
        self.codebook_splits = codebook_splits
        self.separate_codebooks = separate_codebooks
        self.input_size = self.base_encoder.input_size
        self.output_size = self.base_encoder.output_size

        # Loss settings
        self.decoder_encoder_ste = decoder_encoder_ste
        self.use_commitment_loss = use_commitment_loss
        self.vq_loss = vq_loss
        self.embedding_loss_weight = embedding_loss_weight
        self.additional_loss_objective = torch.tensor(0.0)

        # Model
        self.quantizer = self._construct_quantizer()

        if verbose:
            print("Encoder model:\n")
            summary(self, input_size=torch.Size([1,
                                                 self.quantization_heads,
                                                 *self.base_encoder.input_size]))

    def _construct_quantizer(self) -> nn.Module:
        """
        Constructs the quantization layer

        Parameters
        ----------
        None

        Returns
        -------
        model : nn.Modul
            The quantization layer
        """
        if isinstance(self.output_size, int):
            in_out_shape = torch.Size([self.output_size])
        else:
            in_out_shape = self.output_size

        quantizer = VectorQuantizer(codebook_size=self.codebook_size,
                                    in_out_shape=in_out_shape,
                                    codebook_splits=self.codebook_splits,
                                    separate_codebooks=self.separate_codebooks,
                                    st_loss_propagation=self.decoder_encoder_ste,
                                    use_commitment_loss=self.use_commitment_loss,
                                    loss_algorithm=self.vq_loss,
                                    embedding_loss_weight=self.embedding_loss_weight)

        return quantizer

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        z : torch.Tensor
            The output tensor
        loss : torch.Tensor
            The loss
        """
        # Pass through base encoder
        z_1 = self.base_encoder(x)
        # Pass through VQ layer
        z_2 = self.quantizer(z_1)

        return z_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the variational encoder

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        z : torch.Tensor
            The output tensor
        """
        z = self.encode(x)
        self.additional_loss_objective = self.quantizer.get_additional_loss_objective()

        return z

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the latent space

        Parameters
        ----------
        num_samples : int
            The number of samples to generate

        Returns
        -------
        z : torch.Tensor
            The generated samples
        """
        z = self.quantizer.sample(num_samples)

        return z

    def get_additional_loss_objective(self) -> torch.Tensor:
        """
        Get the additional loss objective

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            The additional loss objective
        """
        return self.additional_loss_objective
