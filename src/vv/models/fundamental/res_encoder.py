import typing

import torch
import torch.nn as nn
from torchinfo import summary

from vv.models.core import BaseEncoder
from vv.special_layers import ResidualBlock


class ResEncoder(BaseEncoder):
    """
    Encoder model with a configurable number of convolutional layers, dropout probability and batch normalization
    ----------

    Attributes
    ----------
    input_size: Tuple[int, int, int]
        The size of the input image
    input_units: int
        The number of input channels
    hidden_units: int
        The number of channels for the convolutional layers
    output_units: int
        The number of channels for the output layer
    batch_norm: bool
        Whether to use batch normalization
    dropout_prob: float
        The dropout probability
    layers: nn.Sequential
        The layers in the encoder
    example_input_size: torch.Size
        An examplatory input size
    heads: int
        The number of heads
    parallel: bool
        Whether to use parallel processing or sequential processing for the heads
    """
    def __init__(self,
                 input_size: typing.Tuple[int, int, int],
                 output_units: int = 10,
                 dropout_prob: float = 0.0,
                 batch_norm: bool = False,
                 verbose: bool = False) -> None:
        """
        Encoder model with a configurable number of layers,dropout probability and batch normalization

        Parameters
        ----------
        input_size: Tuple[int, int, int]
            The size of the input image
        output_units : int
            The number of channels for the output layer
        dropout_prob : float
            The dropout probability
        batch_norm : bool
            Whether to use batch normalization
        verbose : bool
            Whether to print the model summary
        """
        super(ResEncoder, self).__init__()

        self.input_size = input_size
        self.input_units = input_size[0]
        self.output_units = output_units
        self.dropout_prob = dropout_prob
        self.batch_norm = batch_norm

        # Conv layer parameters
        self.stride = 2
        self.kernel_size = 4
        self.padding = 3
        self.hidden_units = 256

        # Create an example input tensor
        self.example_input_size = torch.Size([1,
                                              self.input_units,
                                              self.input_size[1],
                                              self.input_size[2]])

        # Construct the encoder
        self.encoder = self._construct_encoder()

        self.output_size = self._calculate_output_size()

        if verbose:
            print("Encoder model:\n")
            print(f"Image size chosen as {self.example_input_size[2]}x{self.example_input_size[3]}")
            summary(self, input_size=self.example_input_size)

    def _construct_encoder(self) -> nn.Sequential:
        """
        Constructs the encoder

        Parameters
        ----------
        None

        Returns
        -------
        model : nn.Sequential
            The encoder model
        """
        layers = []

        layers.append(nn.Conv2d(self.input_units,
                                self.hidden_units,
                                stride=self.stride,
                                kernel_size=self.kernel_size,
                                padding=self.padding))

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_units))
        layers.append(nn.ReLU())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout2d(self.dropout_prob))

        layers.append(nn.Conv2d(self.hidden_units,
                                self.hidden_units,
                                stride=self.stride,
                                kernel_size=self.kernel_size,
                                padding=self.padding))

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_units))
        layers.append(nn.ReLU())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout2d(self.dropout_prob))

        layers.append(ResidualBlock(in_out_units=self.hidden_units,
                                    hidden_units=self.hidden_units,
                                    layer_num=2,
                                    batch_norm=self.batch_norm,
                                    dropout_probability=self.dropout_prob))
        layers.append(ResidualBlock(in_out_units=self.hidden_units,
                                    hidden_units=self.hidden_units,
                                    layer_num=2,
                                    batch_norm=self.batch_norm,
                                    dropout_probability=self.dropout_prob))

        # Reduce channel size with 1 x 1 convolution
        layers.append(nn.Conv2d(self.hidden_units,
                                self.output_units,
                                stride=1,
                                kernel_size=1,
                                padding=0))

        return nn.Sequential(*layers)

    def _calculate_output_size(self) -> tuple:
        """
        Calculate the output size of the encoder

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            The output size
        """
        x = torch.randn(*self.example_input_size)
        z = self.forward(x)
        return tuple(z.size()[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        z : torch.Tensor
            The output tensor
        """
        z = self.encoder(x)

        return z

    def get_intermediate_sizes(self, x: torch.Tensor) -> list[torch.Size]:
        """
        Get the intermediate sizes of the encoder

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        list[torch.Size]
            The intermediate sizes
        """
        sizes = [x.size()]
        for layer in self.encoder:
            x = layer(x)
            sizes.append(x.size())
        return sizes

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
        return torch.tensor(0.0)
