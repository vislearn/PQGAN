import typing

import torch
import torch.nn as nn
from torchinfo import summary

from vv.models.core import BaseEncoder
from vv.special_layers import ResidualBlock


class ResDecoder(BaseEncoder):
    """
    Encoder model with a configurable number of convolutional layers, dropout probability and batch normalization
    ----------

    Attributes
    ----------
    output_channels: int
        The number of input channels
    input_units: int
        The number of input channels
    hidden_units: int
        The number of channels for the convolutional layers
    batch_norm: bool
        Whether to use batch normalization
    dropout_prob: float
        The dropout probability
    stride: int
        The stride for the convolutional layers
    kernel_size: int
        The kernel size for the convolutional layers
    padding: int
        The padding for the convolutional layers
    layers: nn.Sequential
        The layers in the encoder
    example_input_size: torch.Size
        An examplatory input size
    """
    def __init__(self,
                 output_size: typing.Tuple[int, int, int],
                 input_units: int = 10,
                 dropout_prob: float = 0.0,
                 batch_norm: bool = False,
                 verbose: bool = False) -> None:
        """
        Encoder model with a configurable number of layers,dropout probability and batch normalization

        Parameters
        ----------
        output_size: Tuple[int, int, int]
            The size of the output image
        input_units: int
            The number of input channels
        output_channels: int
            The number of output channels
        hidden_units : int
            The number of channels for the convolutional layers
        dropout_prob : float
            The dropout probability
        batch_norm : bool
            Whether to use batch normalization
        verbose : bool
            Whether to print the model summary
        """
        super(ResDecoder, self).__init__()

        self.output_size = output_size
        self.output_units = output_size[0]
        self.input_units = input_units
        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob

        # Conv layer parameters
        self.stride = 2
        self.kernel_size = 4
        self.padding = 3
        self.hidden_units = 256

        # Create an example input tensor
        self.example_input_size = self._calcualte_input_size()

        # Construct the encoder
        upconv_padding = self._calculate_upconv_padding()
        self.layers = self._construct_decoder(upconv_padding)

        if verbose:
            print("Encoder model:\n")
            print(f"Image size chosen as {self.example_input_size[2]}x{self.example_input_size[3]}")
            summary(self, input_size=self.example_input_size)

    def _calculate_upconv_padding(self) -> dict[str, list]:
        """
        Calculate the additional padding for the upconvolutional layers needed due to rounding errors

        Parameters
        ----------
        None

        Returns
        -------
        int
            The padding
        """
        upconv_padding = {"layer1": [0, 0], "layer2": [0, 0]}

        in_x, in_y = self.output_size[1:]
        x_1 = (in_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        y_1 = (in_y - self.kernel_size + 2 * self.padding) // self.stride + 1

        if in_x % 2 != 0:
            upconv_padding["layer2"][0] = 1
        if in_y % 2 != 0:
            upconv_padding["layer2"][1] = 1
        if x_1 % 2 != 0:
            upconv_padding["layer1"][0] = 1
        if y_1 % 2 != 0:
            upconv_padding["layer1"][1] = 1

        return upconv_padding

    def _calcualte_input_size(self) -> torch.Size:
        """
        Calculate the input size of the decoder

        Parameters
        ----------
        None

        Returns
        -------
        torch.Size
            The input size
        """
        in_x, in_y = self.output_size[1:]
        x_1 = (in_x - self.kernel_size + 2 * self.padding) // self.stride + 1
        y_1 = (in_y - self.kernel_size + 2 * self.padding) // self.stride + 1

        x_2 = (x_1 - self.kernel_size + 2 * self.padding) // self.stride + 1
        y_2 = (y_1 - self.kernel_size + 2 * self.padding) // self.stride + 1

        return torch.Size([1, self.hidden_units, x_2, y_2])

    def _construct_decoder(self,
                           upconf_padding: dict[str, list]) -> nn.Sequential:
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

        # Increase channel size with 1 x 1 convolution
        layers.append(nn.Conv2d(self.input_units,
                                self.hidden_units,
                                stride=1,
                                kernel_size=1,
                                padding=0))

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

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_units))
        layers.append(nn.ReLU())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout2d(self.dropout_prob))

        layers.append(nn.ConvTranspose2d(self.hidden_units,
                                         self.hidden_units,
                                         stride=self.stride,
                                         kernel_size=self.kernel_size,
                                         padding=self.padding,
                                         output_padding=upconf_padding["layer1"]))

        if self.batch_norm:
            layers.append(nn.BatchNorm2d(self.hidden_units))
        layers.append(nn.ReLU())
        if self.dropout_prob > 0:
            layers.append(nn.Dropout2d(self.dropout_prob))

        layers.append(nn.ConvTranspose2d(self.hidden_units,
                                         self.output_units,
                                         stride=self.stride,
                                         kernel_size=self.kernel_size,
                                         padding=self.padding,
                                         output_padding=upconf_padding["layer2"]))

        return nn.Sequential(*layers)

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
        z = self.layers(x)
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
        for layer in self.layers:
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
