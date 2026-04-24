from abc import ABC

import torch


class BaseModel(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        pass

    def get_number_of_parameters(self) -> int:
        """
        Returns the total number of parameters in the model

        Parameters
        ----------
        None

        Returns
        -------
        int
            The total number of parameters in the model
        """
        return sum(p.numel() for p in self.parameters())
