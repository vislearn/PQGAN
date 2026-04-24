import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vv.models.latent_diffusion_copy.vector_quantizer import VectorQuantizer2
from vv.shared.shared_types import CodebookUsage, CodebookUsageEntry


class SplitVectorQuantiser(VectorQuantizer2):
    def __init__(self,
                 n_e,
                 e_dim,
                 beta,
                 num_spaces=1,
                 distance_type: str = "l2",
                 analyze_codebook_usage: bool = False,
                 disable_quantization: bool = False):
        assert e_dim % num_spaces == 0, "e_dim must be divisible by num_spaces"
        super().__init__(n_e, e_dim, beta)
        self.num_spaces = num_spaces
        self.distance_type = distance_type # l2 or cosine similarity
        self.analyze_codebook_usage = analyze_codebook_usage
        self.disable_quantization = disable_quantization
        print("If SplitVectorQuantiser used, it is used with distance type:", self.distance_type)
        self.e_dim = e_dim // num_spaces
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_e, self.e_dim) for _ in range(num_spaces)
            ])

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        if self.disable_quantization:
            # If quantization is disabled, return input as is with zero loss and no indices
            loss = torch.tensor(0.0, device=z.device)
            perplexity = None
            min_encodings = None
            min_encoding_indices = None
            return z, loss, (perplexity, min_encodings, min_encoding_indices)

        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim * self.num_spaces)    # [b h w c -> b*h*w, c]

        distances = []
        min_encoding_indices = []
        z_qs = []
        for num, emb in enumerate(self.embeddings):
            z_slice = z_flattened[..., num * self.e_dim:(num + 1) * self.e_dim]  # shape [N, D], N = b*h*w, D = e_dim
            emb_weight = emb.weight  # shape [n_e, D], n_e = number of embeddings, D = e_dim

            if self.distance_type == "l2":
                # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
                d = torch.sum(z_slice ** 2, dim=1, keepdim=True) + \
                    torch.sum(emb_weight**2, dim=1) + \
                    - 2 * torch.einsum('bd,dn->bn', z_slice, rearrange(emb_weight, 'n d -> d n'))

            elif self.distance_type == "cosine":
                z_norm = F.normalize(z_slice, p=2, dim=1)
                e_norm = F.normalize(emb_weight, p=2, dim=1)
                cosine_sim = torch.matmul(z_norm, e_norm.t())
                d = -cosine_sim  # negate for argmin to work as max cosine similarity

            else:
                raise ValueError(f"Unsupported distance_type: {self.distance_type}")

            distances.append(d)
            min_encoding_indices.append(torch.argmin(d, dim=1))
            z_qs.append(
                emb(min_encoding_indices[-1]).view(z[..., num * self.e_dim:(num + 1) * self.e_dim].shape)
            )

        # Analyze codebook utilization
        if self.analyze_codebook_usage:
            self._analysze_codebook_usage(min_encoding_indices)

        perplexity = None
        min_encodings = None
        z_q = torch.cat(z_qs, dim=-1)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def _analysze_codebook_usage(self,
                                 min_encoding_indices: list[torch.Tensor],):
        """
        Analyze the codebook usage and log the results.

        Parameters
        ----------
        trainer : lightning.Trainer
            The lightning trainer instance.
        pl_module : lightning.LightningModule
            The lightning module instance.

        Returns
        ----------
        None
        """
        self.codebook_usage: CodebookUsage = []
        for i, _ in enumerate(min_encoding_indices):
            codebook_statistic = CodebookUsageEntry(
                codebook_index=i,
                bincount=torch.bincount(min_encoding_indices[i], minlength=self.n_e)
            )
            self.codebook_usage.append(codebook_statistic)
