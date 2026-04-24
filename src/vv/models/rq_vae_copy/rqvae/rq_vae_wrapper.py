from contextlib import contextmanager

import lightning as L
import torch

from vv.models.latent_diffusion_copy.util import instantiate_from_config
from vv.models.rq_vae_copy.rqvae.rqvae import RQVAE


class VQModel(L.LightningModule):
    def __init__(self,
                 rq_config,
                 ):
        super().__init__()

        self.automatic_optimization = False

        # instantiate_from_config expects a dict or OmegaConf object
        if not isinstance(rq_config, (dict)):
             raise TypeError(f"rq_config must be a dict, but got {type(rq_config)}")

        self.rq_model = instantiate_from_config(rq_config)

        self.lr_g_factor = 1.0

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=False, **kwargs):
        """
        Custom load_from_checkpoint to handle non-Lightning nn.Module checkpoints.
        """
        # 1. Load the raw state_dict from the non-lightning checkpoint
        sd = torch.load(checkpoint_path, map_location=map_location)

        # The state_dict might be nested under a 'state_dict' key
        if "state_dict" in sd:
            sd = sd["state_dict"]

        # 2. Instantiate the VQModel wrapper, passing init arguments
        model = cls(**kwargs)

        # 3. Load the state_dict into the `rq_model` submodule
        missing, unexpected = model.rq_model.load_state_dict(sd, strict=strict)

        print(f"Restored from {checkpoint_path} with {len(missing)} missing and {len(unexpected)} unexpected keys into rq_model.")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

        # 4. Return the initialized model
        return model

    def init_from_ckpt(self, path, ignore_keys=[]):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        out = self.rq_model.encode(x)
        return out

    def decode(self, quant):
        out = self.rq_model.decode(quant)
        return out

    def forward(self, input, return_pred_indices=False):
        out, quant_loss, code = self.rq_model(input)
        return out, quant_loss, code

    def get_input(self, batch):
        # index 0 is the image, index 1 is the label
        x = batch[0]
        return x

    def training_step(self, batch):
        pass

    def validation_step(self, batch, batch_idx):
        log_dict, rec = self._validation_step(batch, batch_idx)

        result = {"log_dict": log_dict, "rec": rec}

        return result

    def _validation_step(self,
                         batch,
                         batch_idx):
        x = self.get_input(batch)

        xrec, qloss, ind = self(x, return_pred_indices=True)

        return self.log_dict, xrec

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []
