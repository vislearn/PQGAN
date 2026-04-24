import os
import unittest

import torch
from torch import nn

from vv.models.fundamental import ResEncoder, VQEncoder

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestVQEncoder(unittest.TestCase):
    """
    Test class for the vector quantization encoder class
    """
    def setUp(self) -> None:
        """
        Setup the test class
        """
        self.codebook_splits = torch.Size([2, 1, 1])
        self.separate_codebooks = [True, False, False]

        self.res_base_encoder = ResEncoder(input_size=[3, 32, 32],
                                           output_units=10,
                                           dropout_prob=0.0,
                                           batch_norm=True,
                                           verbose=False)

        self.res_1 = VQEncoder(base_encoder=self.res_base_encoder,
                               codebook_splits=self.codebook_splits,
                               separate_codebooks=self.separate_codebooks,
                               codebook_size=10,
                               decoder_encoder_ste=False,
                               use_commitment_loss=True,
                               vq_loss="mse",
                               embedding_loss_weight=0.1,
                               verbose=False)
        self.res_2 = VQEncoder(base_encoder=self.res_base_encoder,
                               codebook_splits=self.codebook_splits,
                               separate_codebooks=self.separate_codebooks,
                               codebook_size=10,
                               decoder_encoder_ste=True,
                               use_commitment_loss=False,
                               vq_loss="mse",
                               embedding_loss_weight=0.1,
                               verbose=False)
        self.res_3 = VQEncoder(base_encoder=self.res_base_encoder,
                               codebook_splits=self.codebook_splits,
                               separate_codebooks=self.separate_codebooks,
                               codebook_size=10,
                               decoder_encoder_ste=False,
                               use_commitment_loss=True,
                               vq_loss="mse",
                               embedding_loss_weight=0.1,
                               verbose=False)

    def test_init(self) -> None:
        """
        Test the initialization of the variational encoder class
        """

        # Check if the model is an instance of variational encoder
        self.assertIsInstance(self.res_1, VQEncoder)
        self.assertIsInstance(self.res_2, VQEncoder)
        self.assertIsInstance(self.res_3, VQEncoder)
        self.assertIsInstance(self.res_1, nn.Module)
        self.assertIsInstance(self.res_2, nn.Module)
        self.assertIsInstance(self.res_3, nn.Module)

    def test_encode(self) -> None:
        """
        Test the encode method of the variational encoder class
        """

        # Generate random input data
        x = torch.randn(32, 3, 32, 32)

        # Pass the input data through the model
        z_lin = self.res_1.encode(x)
        z_conv = self.res_2.encode(x)
        z_res = self.res_3.encode(x)

        # Check the shape of the output
        self.assertEqual(z_lin.shape[0], x.shape[0])
        self.assertEqual(z_conv.shape[0], x.shape[0])
        self.assertEqual(z_res.shape[0], x.shape[0])

        print(self.res_1.output_size)

        self.assertEqual(z_lin.shape[1:], torch.Size([*self.res_1.output_size]))
        self.assertEqual(z_conv.shape[1:], torch.Size([*self.res_2.output_size]))
        self.assertEqual(z_res.shape[1:], torch.Size([*self.res_3.output_size]))

    def test_forward(self) -> None:
        """
        Test the forward pass of the variational encoder class
        """

        # Generate random input data
        x = torch.randn(32, 3, 32, 32)

        # Pass the input data through the model
        z_lin = self.res_1.forward(x)
        z_conv = self.res_2.forward(x)
        z_res = self.res_3.forward(x)

        # Check the shape of the output
        self.assertEqual(z_lin.shape[0], x.shape[0])
        self.assertEqual(z_conv.shape[0], x.shape[0])
        self.assertEqual(z_res.shape[0], x.shape[0])
        self.assertEqual(z_lin.shape[1:], torch.Size([*self.res_1.output_size]))
        self.assertEqual(z_conv.shape[1:], torch.Size([*self.res_2.output_size]))
        self.assertEqual(z_res.shape[1:], torch.Size([*self.res_3.output_size]))

    def test_sample(self) -> None:
        """
        Test the sample method of the variational encoder class
        """

        # Generate random input data
        n = 32

        # Pass the input data through the model
        z_lin = self.res_1.sample(n)
        z_conv = self.res_2.sample(n)
        z_res = self.res_3.sample(n)

        # Check the shape of the output
        self.assertEqual(z_lin.shape[0], n)
        self.assertEqual(z_conv.shape[0], n)
        self.assertEqual(z_res.shape[0], n)
        self.assertEqual(z_lin.shape[1:], torch.Size([*self.res_1.output_size]))
        self.assertEqual(z_conv.shape[1:], torch.Size([*self.res_2.output_size]))
        self.assertEqual(z_res.shape[1:], torch.Size([*self.res_3.output_size]))

    def test_get_additional_loss_objective(self) -> None:
        """
        Test the get additional loss objective method
        """
        # Get the additional loss objective
        lin_loss = self.res_1.get_additional_loss_objective()
        conv_loss = self.res_2.get_additional_loss_objective()

        self.assertEqual(lin_loss, torch.tensor(0.0))
        self.assertEqual(conv_loss, torch.tensor(0.0))

        # Generate random input data
        x = torch.randn(32, 3, 32, 32)

        # Pass the input data through the model
        _ = self.res_1.forward(x)
        _ = self.res_2.forward(x)

        # Get the additional loss objective
        lin_loss = self.res_1.get_additional_loss_objective()
        conv_loss = self.res_2.get_additional_loss_objective()

        self.assertNotEqual(lin_loss, torch.tensor(0.0))
        self.assertNotEqual(conv_loss, torch.tensor(0.0))

        # Check the shape of the output
        self.assertEqual(lin_loss.size(), ())
        self.assertEqual(conv_loss.size(), ())

        # Check the type of the output
        self.assertIsInstance(lin_loss, torch.Tensor)
        self.assertIsInstance(conv_loss, torch.Tensor)
