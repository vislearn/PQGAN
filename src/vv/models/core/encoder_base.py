from abc import abstractmethod

import torch

from .model_base import BaseModel


class BaseEncoder(BaseModel):
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()

        # Needs to have member input_size: typing.Tuple[int, int, int],
        # Needs to have member output_size: int

    @abstractmethod
    def get_additional_loss_objective(self) -> torch.Tensor:
        """
        Function that should return any additional loss objective that the encoder may have

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            The additional loss objective
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
