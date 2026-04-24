import torch


class DummyLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # Return zero loss and a log dictionary with expected keys
        log = {
            f"{kwargs.get('split', 'val')}/total_loss": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/quant_loss": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/nll_loss": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/rec_loss": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/p_loss": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/d_weight": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/disc_factor": torch.tensor(0.0),
            f"{kwargs.get('split', 'val')}/g_loss": torch.tensor(0.0),
        }
        return torch.tensor(0.0), log
