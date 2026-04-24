import torch
# Callbacks for Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # noqa: F401
# Logger
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: F401

# Custom data and models
import vv.data as data  # noqa: F401
# LightningCLI
import vv.execution as execution
import vv.models as models  # noqa: F401
import vv.models.fundamental as fundamental  # noqa: F401


def cli_main(argv: str = None) -> None:
    # Set the precision for matrix multiplications
    torch.set_float32_matmul_precision('medium')
    cli = execution.CustomCLI()  # noqa: F841
