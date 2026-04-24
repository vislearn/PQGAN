import os
import unittest

import torch

from vv.special_layers import VectorQuantizer

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestVQLayer(unittest.TestCase):
    """
    Test class for the vector quantization layer
    """
    def setUp(self) -> None:
        """
        Set up the test
        """
        self.input_dim_flat = torch.Size([6])
        self.flat_splits = torch.Size([2])
        self.flat_seperate_codebooks = [True]
        self.input_dim_image = torch.Size([1, 4, 3])
        self.image_splits = torch.Size([1, 2, 1])
        self.image_seperate_codebooks = [True, False, True]
        self.embedding_size = 3
        self.embedding_loss_weight = 0.25

        self.vq_flat = VectorQuantizer(codebook_size=self.embedding_size,
                                       in_out_shape=self.input_dim_flat,
                                       codebook_splits=self.flat_splits,
                                       separate_codebooks=self.flat_seperate_codebooks,
                                       st_loss_propagation=False,
                                       use_commitment_loss=True,
                                       loss_algorithm="mse",
                                       embedding_loss_weight=self.embedding_loss_weight)

        self.vq_img = VectorQuantizer(codebook_size=self.embedding_size,
                                      in_out_shape=self.input_dim_image,
                                      codebook_splits=self.image_splits,
                                      separate_codebooks=self.image_seperate_codebooks,
                                      st_loss_propagation=True,
                                      use_commitment_loss=True,
                                      loss_algorithm="mse",
                                      embedding_loss_weight=self.embedding_loss_weight)

        self.vq_channels = VectorQuantizer(codebook_size=self.embedding_size,
                                           in_out_shape=self.input_dim_image,
                                           codebook_splits=self.image_splits,
                                           separate_codebooks=self.image_seperate_codebooks,
                                           st_loss_propagation=True,
                                           use_commitment_loss=False,
                                           loss_algorithm="mse",
                                           embedding_loss_weight=self.embedding_loss_weight)

    def test_init(self) -> None:
        """
        Test the initialization of the autoencoder class
        """
        self.assertIsInstance(self.vq_flat, VectorQuantizer)
        self.assertIsInstance(self.vq_img, VectorQuantizer)
        self.assertIsInstance(self.vq_channels, VectorQuantizer)
        self.assertEqual(type(self.vq_flat.embedding_space), torch.nn.Parameter)
        self.assertEqual(type(self.vq_img.embedding_space), torch.nn.Parameter)
        self.assertEqual(type(self.vq_channels.embedding_space), torch.nn.Parameter)

        self.assertEqual(self.vq_flat.embedding_space.size(), torch.Size([1,
                                                                          self.embedding_size,
                                                                          *self.flat_splits,
                                                                          3]))
        image_heads = [split if head else 1 for split, head in zip(self.image_splits, self.image_seperate_codebooks)]
        self.assertEqual(self.vq_img.embedding_space.size(), torch.Size([1,
                                                                         self.embedding_size,
                                                                         *image_heads,
                                                                         1,
                                                                         2,
                                                                         3]))
        self.assertEqual(self.vq_channels.embedding_space.size(), torch.Size([1,
                                                                              self.embedding_size,
                                                                              *image_heads,
                                                                              1,
                                                                              2,
                                                                              3]))

        # Check the range of embedding weights
        min_value_flat = -1
        max_value_flat = 1
        min_value_img = -1
        max_value_img = 1

        flat_embedding_space = self.vq_flat.embedding_space
        img_embedding_space = self.vq_img.embedding_space

        self.assertTrue(torch.all(flat_embedding_space >= min_value_flat))
        self.assertTrue(torch.all(flat_embedding_space <= max_value_flat))
        self.assertTrue(torch.all(img_embedding_space >= min_value_img))
        self.assertTrue(torch.all(img_embedding_space <= max_value_img))

    def test_forward(self) -> None:
        """
        Test the forward pass of the layer and mainly focus on the shape of the output
        """
        # Should work
        flat_size = torch.Size([1, *self.input_dim_flat])
        img_size = torch.Size([1, *self.input_dim_image])

        x_flat = torch.rand(*flat_size)
        x_img = torch.rand(*img_size)

        quantized_flat = self.vq_flat.forward(x_flat)
        quantized_img = self.vq_img.forward(x_img)
        quantized_channels = self.vq_channels.forward(x_img)

        self.assertEqual(quantized_flat.size(), flat_size)
        self.assertEqual(quantized_img.size(), img_size)
        self.assertEqual(quantized_channels.size(), img_size)

        # Should raise an error
        x_3 = torch.randn(3)
        x_4 = torch.randn(3, 3)
        x_5 = torch.randn(3, 3, 3)

        with self.assertRaises(RuntimeError):
            self.vq_flat.forward(x_3)
        with self.assertRaises(RuntimeError):
            self.vq_img.forward(x_3)
        with self.assertRaises(RuntimeError):
            self.vq_channels.forward(x_3)
        with self.assertRaises(RuntimeError):
            self.vq_flat.forward(x_4)
        with self.assertRaises(RuntimeError):
            self.vq_img.forward(x_4)
        with self.assertRaises(RuntimeError):
            self.vq_channels.forward(x_4)
        with self.assertRaises(RuntimeError):
            self.vq_flat.forward(x_5)
        with self.assertRaises(RuntimeError):
            self.vq_img.forward(x_5)
        with self.assertRaises(RuntimeError):
            self.vq_channels.forward(x_5)

    def test_forward_k_search(self) -> None:
        """
        Test if the input is mapped to the correct embedding
        """
        x = torch.rand(1, *self.input_dim_image)

        quantized = self.vq_img.forward(x)

        print(x.shape)
        print(quantized.shape)

        print(x)
        print(quantized)
        print(self.vq_img.embedding_space)

        print(sum(p.numel() for p in self.vq_img.parameters()))

        # Check manually
        self.assertTrue(True)

    def test_backward(self) -> None:
        """
        Test the backward pass of the layer
        """
        x = torch.randn(5, *self.input_dim_image, requires_grad=True)
        _ = self.vq_img.forward(x)
        loss = self.vq_img.get_additional_loss_objective()
        loss.backward()

        self.assertIsNotNone(self.vq_img.embedding_space.grad)
        image_heads = [split if head else 1 for split, head in zip(self.image_splits, self.image_seperate_codebooks)]
        self.assertEqual(self.vq_img.embedding_space.grad.size(), torch.Size([1,
                                                                              self.embedding_size,
                                                                              *image_heads,
                                                                              1,
                                                                              2,
                                                                              3]))
        self.assertIsNotNone(self.vq_img.embedding_space.grad)
        self.assertEqual(self.vq_img.embedding_space.grad.size(), torch.Size([1,
                                                                              self.embedding_size,
                                                                              *image_heads,
                                                                              1,
                                                                              2,
                                                                              3]))

    def test_get_additional_loss_objective(self) -> None:
        """
        Test the calculation of the additional loss objective
        """
        x = torch.randn(5, *self.input_dim_image, requires_grad=True)
        _ = self.vq_img.forward(x)
        loss = self.vq_img.get_additional_loss_objective()

        self.assertEqual(loss.size(), torch.Size([]))
        self.assertEqual(loss.requires_grad, True)

        loss.backward()

    def test_sample(self) -> None:
        """
        Test the sampling of the layer
        """
        sample_flat = self.vq_flat.sample(5)
        sample_img = self.vq_img.sample(5)
        smple_channels = self.vq_channels.sample(5)

        self.assertEqual(sample_flat.size(), torch.Size([5, *self.input_dim_flat]))
        self.assertEqual(sample_img.size(), torch.Size([5, *self.input_dim_image]))
        self.assertEqual(smple_channels.size(), torch.Size([5, *self.input_dim_image]))
