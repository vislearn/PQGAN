import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block, input size has to be 256 channels

    ----------
    Attributes
    ----------
    in_out_units: int
        The number of channels for the input and output convolutional layers
    hidden_units: int
        The number of channels for the convolutional layers
    layer_num: int
        The number of layers in the block
    layers: nn.Sequential
        The layers in the block
    """
    def __init__(self,
                 in_out_units: int = 256,
                 hidden_units: int = 256,
                 layer_num: int = 2,
                 batch_norm: bool = True,
                 dropout_probability: float = 0.0) -> None:
        """
        Initialize the variational layer

        Parameters
        ----------
        in_out_units: int
            The number of channels for the input and output convolutional layers
        hidden_units: int
            The number of channels for the convolutional layers
        layer_num: int
            The number of layers in the block
        batch_norm: bool
            Whether to use batch normalization
        dropout_probability: float
            The dropout probability
        """
        super(ResidualBlock, self).__init__()

        self.in_out_units = in_out_units
        self.hidden_units = hidden_units
        self.layer_num = layer_num
        self.batch_norm = batch_norm
        self.dropout_probability = dropout_probability

        self.layers = self._construct_block()

    def _construct_block(self) -> nn.Sequential:
        """
        Construct the residual block
        """
        block = []

        if self.batch_norm:
            block.append(nn.BatchNorm2d(self.in_out_units))

        block.append(nn.ReLU())

        if self.dropout_probability > 0:
            block.append(nn.Dropout2d(self.dropout_probability))

        block.append(nn.Conv2d(self.in_out_units,
                               self.hidden_units,
                               stride=1,
                               kernel_size=3,
                               padding=1))

        for _ in range(self.layer_num - 2):
            if self.batch_norm:
                block.append(nn.BatchNorm2d(self.hidden_units))

            block.append(nn.ReLU())

            if self.dropout_probability > 0:
                block.append(nn.Dropout2d(self.dropout_probability))

            block.append(nn.Conv2d(self.hidden_units,
                                   self.hidden_units,
                                   stride=1,
                                   kernel_size=3,
                                   padding=1))

        if self.batch_norm:
            block.append(nn.BatchNorm2d(self.hidden_units))

        block.append(nn.ReLU())

        if self.dropout_probability > 0:
            block.append(nn.Dropout2d(self.dropout_probability))

        block.append(nn.Conv2d(self.hidden_units,
                               self.in_out_units,
                               stride=1,
                               kernel_size=1,
                               padding=0))

        return nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        z: torch.Tensor
            The output tensor
        """
        z = x + self.layers(x)

        return z
