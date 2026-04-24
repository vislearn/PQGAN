# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding models used in the CMMD calculation."""

import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"


class ClipEmbeddingModel(nn.Module):
    """CLIP image embedding calculator."""

    def __init__(self) -> None:
        super().__init__()
        self._model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME)  # takes (b, h, w, c)

        # Register buffers so they move with the model across devices
        # https://github.com/openai/CLIP/blob/main/clip/clip.py
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.clip_image_size = 336  # It's in the name...

    def _resize_bicubic(self,
                        images: torch.Tensor) -> torch.Tensor:
        """Resizes images using bicubic interpolation.
        Args:
            images: An image tensor of shape (b, 3, h, w). Values are in range [0, 1].
        Returns:
            Resized image tensor of shape (b, 3, size, size).
        """
        images = torch.nn.functional.interpolate(images, size=(self.clip_image_size, self.clip_image_size), mode="bicubic")
        return images

    def _normalize_clip(self,
                        images: torch.Tensor) -> torch.Tensor:
        """
        Normalizes images using the CLIP mean and std.
        Args:
            images: An image tensor of shape (b, 3, h, w). Values are in range [0, 1].
        Returns:
            Normalized image tensor of shape (b, 3, h, w).
        """
        return (images - self.clip_mean) / self.clip_std  # type: ignore

    def _preprocess(self,
                    images: torch.Tensor) -> torch.Tensor:
        """Preprocesses the images for CLIP.

        Args:
            images: An image tensor of shape (batch_size, 3, height, width). Values are in range [0, 1].

        Returns:
            Preprocessed image tensor of shape (batch_size, 3, height, width).
        """
        images = self._resize_bicubic(images)
        images = self._normalize_clip(images)
        return images

    def embed(self,
              images: torch.Tensor) -> torch.Tensor:
        """Computes CLIP embeddings for the given images.

        Args:
          images: An image tensor of shape (batch_size, 3, height, width). Values are
            in range [0, 1].

        Returns:
          Embedding array of shape (batch_size, embedding_width).
        """
        images = self._preprocess(images)

        image_embs = self._model(images).image_embeds
        image_embs /= torch.linalg.norm(image_embs, axis=-1, keepdims=True)
        return image_embs
