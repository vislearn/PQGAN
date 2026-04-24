import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector quantization layer. It quantizes the input to a finite set of embeddings

    ----------
    Attributes
    ----------
    quantization_heads: int
        The number of attention heads
    embedding_shape: int
        The number of embeddings
    individual_pixel_embedding: bool
        Whether to use individual pixel embeddings. If False, the same embedding is used for all pixels (channels still individual)
    in_out_shape: torch.Size
        The dimensionality of input and the output
    embedding_shape: typing.Tuple
        The shape of the embeddings
    st_loss_propagation: bool
        Whether to use the straight through estimator from the decoder to the encoder
    use_commitment_loss: bool
            Whether to use the commitment loss for the encoder to the embedding
    loss_algorithm: str
        The loss algorithm to use for updating the embeddings
    embedding_loss_weight: float
        The commitment cost
    embedding_space: torch.nn.Parameter
        Embeddings stored as a tensor
    loss: torch.Tensor
        The loss
    """
    def __init__(self,
                 codebook_size: int,
                 in_out_shape: torch.Size,
                 codebook_splits: typing.Tuple[int, ...],
                 separate_codebooks: typing.Tuple[bool, ...],
                 st_loss_propagation: bool,
                 use_commitment_loss: bool,
                 loss_algorithm: str,
                 embedding_loss_weight: int) -> None:
        """
        Initilized embeddings to uniform values

        Parameters
        ----------
        codebook_size: int
            The number of embeddings
        in_out_shape: torch.Size
            The dimensionality of the embeddings
        codebook_splits: typing.Tuple[int, ...]
            Splits the respective dimensions into several smaller dimensions. These are considered as vectors
        separate_codebooks: typing.Tuple[bool, ...]
            The number of codebooks to use. Each dimention that is split can have an adjustable number of codebooks
            The number of codebooks is 1 (False) or as many as there are splits (True)
        st_loss_propagation: bool
            Whether to use the straight through estimator from the decoder to the encoder
        use_commitment_loss: bool
            Whether to use the commitment loss for the encoder to the embedding
        loss_algorithm: str
            The loss algorithm to use for updating the embeddings
        embedding_loss_weight: float
            The commitment cost
        """
        super(VectorQuantizer, self).__init__()

        # Embedding space
        self.codebook_size = codebook_size
        self.in_out_shape = in_out_shape
        self.codebook_splits = codebook_splits
        self.separate_codebooks = separate_codebooks

        assert len(self.in_out_shape) == len(self.codebook_splits), \
            'The number of splits must match the number of dimensions'
        assert len(self.in_out_shape) == len(self.separate_codebooks), \
            'separate_codebooks should have the same length as in_out_shape'

        self.embedding_space, self.embedding_dimensions, self.head_count = self._construct_embedding_space()

        # Loss calculation
        self.st_loss_propagation = st_loss_propagation
        self.use_commitment_loss = use_commitment_loss
        self.loss_algorithm = loss_algorithm
        self.embedding_loss_weight = embedding_loss_weight
        self.loss = torch.tensor(0.0)

        # Forward computation
        self.interleaved_view = [dim for pair in zip(self.codebook_splits, self.embedding_dimensions) for dim in pair]
        self.reordeing_scheme, self.inverse_reordering = self._create_reordering_scheme(self.embedding_dimensions)

    def _construct_embedding_space(self) -> typing.Tuple[torch.nn.Parameter, typing.List[int], typing.List[int]]:
        """
        Constructs the embedding space

        Returns
        -------
        embedding_space: torch.nn.Parameter
            The embedding space
        """
        embedding_dimensions = self._calculate_embedding_dimensions()
        head_count = self._calculate_head_count()

        # Values initilized between -1 and 1 and then normalized
        total_embedding_size = torch.prod(torch.tensor(embedding_dimensions))
        embedding_space = nn.Parameter((2 * torch.rand(1,
                                                       self.codebook_size,
                                                       *head_count,
                                                       *embedding_dimensions) - 1) / total_embedding_size)

        return embedding_space, embedding_dimensions, head_count

    def _calculate_embedding_dimensions(self) -> torch.Size:
        """
        Calculates the embedding dimensions

        Returns
        -------
        embedding_dimensions: typing.List[int]
            The embedding dimensions
        """
        # Assert evenly devided splits
        for dim_size, code_book_split in zip(self.in_out_shape, self.codebook_splits):
            assert dim_size % code_book_split == 0, 'The splits must divide the respective dimensions'

        embedding_dimensions = []
        for dimension_size, codebook_split in zip(self.in_out_shape, self.codebook_splits):
            embedding_dimensions.append(int(dimension_size / codebook_split))

        return torch.Size(embedding_dimensions)

    def _calculate_head_count(self) -> typing.List[int]:
        """
        Calculates the number of heads

        Returns
        -------
        head_count: int
            The number of heads
        """
        head_count = []
        for codebook_split, separate_codebook in zip(self.codebook_splits, self.separate_codebooks):
            if separate_codebook:
                head_count.append(codebook_split)
            else:
                head_count.append(1)

        return head_count

    def _create_reordering_scheme(self,
                                  embedding_dimensions: typing.List) -> typing.Tuple[typing.List,
                                                                                     typing.List]:
        """
        Creates a reordering scheme for the embeddings

        Parameters
        ----------
        embedding_dimensions: typing.List
            The embedding dimensions

        Returns
        -------
        reordering: torch.Tensor
            The reordering scheme
        """
        const_dims = 1
        num_dims = 2 * len(embedding_dimensions)
        even_indices = [i + const_dims for i in range(num_dims) if i % 2 != 0]
        odd_indices = [i + const_dims for i in range(num_dims) if i % 2 == 0]
        reordering = [0] + odd_indices + even_indices

        inverse_reordering = [0] * len(reordering)
        for i, p in enumerate(reordering):
            inverse_reordering[p] = i

        return reordering, inverse_reordering

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        quantized: torch.Tensor
            The quantized tensor
        """
        input_shape = x.shape
        batch_size = input_shape[0]

        x_reshaped = self._reshape_input(x)

        embedding_space_expanded_batch = self._reshape_codebook(batch_size, self.embedding_space)

        l2_distances = self._calcualte_l2_distances(x_reshaped, embedding_space_expanded_batch)

        # Find the index of the nearest value
        indices = self._calcualte_indices(l2_distances)

        # Get the corresponding elements
        quantized = self._get_quantizaion_values(indices, embedding_space_expanded_batch, input_shape)

        # Set the loss
        output = self._set_loss_propagation(x, quantized)
        return output

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input tensor and splits it into the respective heads
        Copies the tensor in dim 1 to match the size of the codebook

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        x_expanded: torch.Tensor
            The reshaped tensor
        """
        batch_size = x.shape[0]

        # (batch_size, *in_out_shape) -> (batch_size, head_i, embedding_dim_i, ...)
        x_reshaped = x.view(batch_size, *self.interleaved_view)
        # (batch_size, head_i, ..., embedding_dim_i, ...)
        x_rearranged = x_reshaped.permute(*self.reordeing_scheme)
        # (batch_size, codebook_size, head_i, ..., embedding_dim_i, ...)
        x_rearranged = x_rearranged.unsqueeze(1)
        x_expanded = x_rearranged.expand(batch_size, self.codebook_size, *x_rearranged.shape[2:])

        return x_expanded

    def _reshape_codebook(self,
                          batch_size: int,
                          codebook: nn.Parameter) -> torch.Tensor:
        """
        Reshapes the codebook tensor

        Parameters
        ----------
        codebook: nn.Parameter
            The codebook tensor

        Returns
        -------
        embedding_space_expanded_batch: torch.Tensor
            The reshaped codebook
        """
        # (1, head_i, ..., embedding_dim_i, ...) -> (batch_size, head_i, ..., embedding_dim_i, ...)
        # Copies head to all split dimentions if separate_codebooks is False
        embedding_space_expanded_batch = codebook.expand(batch_size,
                                                         self.codebook_size,
                                                         *self.codebook_splits,
                                                         *self.embedding_dimensions)

        return embedding_space_expanded_batch

    def _calcualte_l2_distances(self,
                                x_reshaped: torch.Tensor,
                                embedding_space_expanded_batch: nn.Parameter) -> torch.Tensor:
        """
        Calculates the L2 distances between the input and the embeddings

        Parameters
        ----------
        x_reshaped: torch.Tensor
            The reshaped input tensor
        embedding_space_expanded_batch: nn.Parameter
            The expanded embedding space

        Returns
        -------
        l2_distances: torch.Tensor
            The L2 distances
        """
        batch_size = x_reshaped.shape[0]

        # (batch_size, codebook_size, head_i, ..., embedding_dim_i, ...)
        distances = torch.abs(x_reshaped - embedding_space_expanded_batch)

        # (batch_size, codebook_size, head_i, ..., flat_embedding_dim)
        total_embedding_size = torch.prod(torch.tensor(self.embedding_dimensions))
        distances_flat_embedding = distances.view(batch_size,
                                                  self.codebook_size,
                                                  *self.codebook_splits,
                                                  total_embedding_size)

        # (batch_size, codebook_size, head_i, ..., flat_embedding_dim) -> (batch_size, codebook_size, head_i, ...)
        l2_distances = torch.norm(distances_flat_embedding, dim=-1)

        return l2_distances

    def _calcualte_indices(self,
                           l2_distances: torch.Tensor) -> torch.Tensor:
        """
        Calculates the indices of the nearest embeddings

        Parameters
        ----------
        l2_distances: torch.Tensor
            The L2 distances

        Returns
        -------
        indices: torch.Tensor
            The indices
        """
        batch_size = l2_distances.shape[0]

        # (batch_size, head_i, head_i+1, ..., 1, ...)
        indices = torch.argmin(l2_distances, dim=1).unsqueeze(1)
        for _ in range(len(self.in_out_shape)):
            indices = indices.unsqueeze(-1)

        # (batch_size, head_i, head_i+1, ..., 1, ...) -> (batch_size, 1, head_i, head_i+1, ..., embedding_dim_i, ...)
        indices_expanded = indices.expand(batch_size,
                                          1,
                                          *self.codebook_splits,
                                          *self.embedding_dimensions)

        return indices_expanded

    def _get_quantizaion_values(self,
                                indices: torch.Tensor,
                                embedding_space_expanded_batch: nn.Parameter,
                                original_shape: torch.Size) -> torch.Tensor:
        """
        Gets the quantized values

        Parameters
        ----------
        indices: torch.Tensor
              The indices

        Returns
        -------
        quantized: torch.Tensor
              The quantized values
        """
        # (batch_size, attention heads, *embedding_dim)
        quantized = torch.gather(embedding_space_expanded_batch, dim=1, index=indices).squeeze(1)
        # Back to original shape
        quantized_permuted = quantized.permute(*self.inverse_reordering)
        quantized_orig_shape = quantized_permuted.reshape(original_shape)

        return quantized_orig_shape

    def _set_loss_propagation(self,
                              x: torch.Tensor,
                              x_quantized: torch.Tensor) -> torch.Tensor:
        """
        Sets the loss propagation

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        x_quantized: torch.Tensor
            The output tensor

        Returns
        -------
        output: torch.Tensor
            The output tensor
        """
        # Reconstruction loss propagation
        if self.st_loss_propagation:
            # Note: In this case the VQ will not receive any reconstruction losses (As in Google paper)
            output = x + (x_quantized - x).detach()
        else:
            output = x_quantized

        # Commitment loss propagation
        if self.use_commitment_loss:
            # Commitment loss for the encoder to the embedding
            e_latent_loss = F.mse_loss(output.detach(), x)
        else:
            e_latent_loss = torch.tensor(0.0, device=self.embedding_space.device)

        # VQ loss propagation
        if self.loss_algorithm == 'mse':
            q_latent_loss = F.mse_loss(x_quantized, x.detach())
        elif self.loss_algorithm == 'ema':
            q_latent_loss = torch.tensor(0.0, device=self.embedding_space.device)
        else:
            raise ValueError('VQ loss not recognized')

        # Total loss
        loss = q_latent_loss + self.embedding_loss_weight * e_latent_loss
        self.loss = loss

        return output

    def get_additional_loss_objective(self) -> torch.Tensor:
        """
        Returns the loss of the layer

        Returns
        -------
        loss: torch.Tensor
            The loss
        """
        return self.loss

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Samples from the embedding space

        Parameters
        ----------
        batch_size: int
            The number of samples to take

        Returns
        -------
        mean_samples: torch.Tensor
            The samples
        """
        # Copy the embedding alsong the embedding dim if individual_pixel_embedding is False
        embedding_space_expanded = self._reshape_codebook(batch_size, self.embedding_space)

        indices = self._generate_random_indices(batch_size).unsqueeze(1)

        # Use these indices to gather elements from embedding_space
        shape = torch.Size([batch_size, *self.in_out_shape])

        samples = self._get_quantizaion_values(indices, embedding_space_expanded, shape)

        return samples

    def _generate_random_indices(self, batch_size: int) -> torch.Tensor:
        """
        Generates random indices

        Parameters
        ----------
        batch_size: int
            The number of samples to take

        Returns
        -------
        indices: torch.Tensor
            The indices
        """
        indices = torch.randint(low=0,
                                high=self.codebook_size,
                                size=(batch_size,
                                      *self.codebook_splits,
                                      *self.embedding_dimensions),
                                device=self.embedding_space.device)

        return indices
