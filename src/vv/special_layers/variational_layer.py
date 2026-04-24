import torch
import torch.nn as nn


class VariationalLayer(nn.Module):
    """
    Variational layer that samples from a normal distribution

    ----------
    Attributes
    ----------
    mu_layer: nn.Linear
        The linear layer for the mean
    logvar_layer: nn.Linear
        The linear layer for the log variance
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int) -> None:
        """
        Initialize the variational layer

        Parameters
        ----------
        input_dim: int
            The input dimension
        output_dim: int
            The output dimension
        """
        super(VariationalLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.logvar_layer = nn.Linear(input_dim, output_dim)

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
            The sampled tensor
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        # Reparameterization trick to sample from the distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar
