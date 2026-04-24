import lightning as L  # noqa: F401
import torch  # noqa: F401
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # noqa: F401
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: F401
from lightning.pytorch.utilities.model_summary import ModelSummary  # noqa: F401

import vv.data as data  # noqa: F401
import vv.models as models  # noqa: F401
import vv.models.fundamental as fm  # noqa: F401
import vv.special_layers as sl  # noqa: F401
from vv.utilities import get_dir  # noqa: F401


def debug_main(argv: str = None) -> None:
    vq_flat = sl.VectorQuantizer(codebook_size=512,
                                 in_out_shape=[10, 11, 11],
                                 codebook_splits=[10, 11, 11],
                                 separate_codebooks=[True, False, False],
                                 st_loss_propagation=False,
                                 use_commitment_loss=True,
                                 loss_algorithm="mse",
                                 embedding_loss_weight=0.25)

    model_parameters = filter(lambda p: p.requires_grad, vq_flat.parameters())
    params_count = sum([torch.numel(p) for p in model_parameters])
    print("test")
    print(f"Number of parameters: {params_count}")
