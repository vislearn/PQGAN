import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """
    Dense block

    ----------
    Attributes
    ----------
    num_layers: int
        The number of layers in the block
    input_channels: int
    """
    def __init__(self,
                 input_channels: int,
                 growth_rate: int = 32,
                 num_layers: int = 3,
                 reduce_channels: bool = False,
                 batch_norm: bool = True,
                 dropout_probability: float = 0.0) -> None:
        """
        Initialize the dense block

        Parameters
        ----------
        input_channels: int
            The number of input channels
        growth_rate: int
            The growth rate of the block
        num_layers: int
            The number of layers in the block
        reduce_channels: bool
            Whether to reduce the number of channels back to the input channels at the end of the block
        batch_norm: bool
            Whether to use batch normalization
        dropout_probability: float
            The dropout probability
        """
        super(DenseBlock, self).__init__()

        self.num_layers = num_layers
        self.input_channels = input_channels
        self.growth_rate = growth_rate
        self.reduce_channels = reduce_channels
        self.batch_norm = batch_norm
        self.dropout_probability = dropout_probability

        self.kernel_size = 3
        self.stide = 1
        self.padding = 1

        # Construct the dense block
        self.layers = self._construct_block()

    def _construct_block(self) -> nn.ModuleList:
        """
        Construct the dense block

        Parameters
        ----------
        None

        Returns
        -------
        nn.ModuleList
            The layers in the block
        """
        layers = []
        for i in range(self.num_layers):
            layer_input_channels = self.input_channels + i * self.growth_rate
            layers.append(self._construct_layer(layer_input_channels))
        if self.reduce_channels:
            final_layer_input_channels = layer_input_channels + self.growth_rate
            layers.append(self._construct_final_layer(final_layer_input_channels))
        return nn.ModuleList(layers)

    def _construct_layer(self,
                         input_channels: int) -> nn.Sequential:
        """
        Construct a single layer in the block

        Parameters
        ----------
        input_channels: int
            The number of input channels

        Returns
        -------
        nn.Sequential
            The layer
        """
        layers = []
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(input_channels))
        layers.append(nn.ReLU(inplace=True))
        if self.dropout_probability > 0:
            layers.append(nn.Dropout2d(self.dropout_probability))
        layers.append(nn.Conv2d(input_channels,
                                self.growth_rate,
                                kernel_size=self.kernel_size,
                                stride=self.stide,
                                padding=self.padding,
                                bias=False))
        return nn.Sequential(*layers)

    def _construct_final_layer(self,
                               input_channels: int) -> nn.Sequential:
        """
        Construct the final layer in the block

        Parameters
        ----------
        input_channels: int
            The number of input channels

        Returns
        -------
        nn.Sequential
            The final layer
        """
        layers = []
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(input_channels))
        layers.append(nn.ReLU(inplace=True))
        if self.dropout_probability > 0:
            layers.append(nn.Dropout2d(self.dropout_probability))
        layers.append(nn.Conv2d(input_channels,
                                self.input_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False))
        return nn.Sequential(*layers)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dense block

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        features = [x]
        # Process all layers except the last if reduce_channels is True, otherwise process all layers
        for layer in self.layers[:-1] if self.reduce_channels else self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)

        # If reduce_channels is True, process the last layer separately
        if self.reduce_channels:
            out = self.layers[-1](torch.cat(features, 1))
            return out
        else:
            return torch.cat(features, 1)
