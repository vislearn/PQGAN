import os
import unittest

import torch

from vv.special_layers import ResidualBlock

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestResidualBlock(unittest.TestCase):
    """
    Test class for the variational layer
    """
    def setUp(self) -> None:
        """
        Set up the test
        """
        self.res_block_1 = ResidualBlock(in_out_units=256,
                                         hidden_units=256,
                                         layer_num=2,
                                         batch_norm=True,
                                         dropout_probability=0.1)
        self.res_block_2 = ResidualBlock(in_out_units=10,
                                         hidden_units=256,
                                         layer_num=10,
                                         batch_norm=True,
                                         dropout_probability=0.0)

    def test_init(self) -> None:
        """
        Test the initialization of the autoencoder class
        """
        self.assertIsInstance(self.res_block_1, ResidualBlock)
        self.assertIsInstance(self.res_block_2, ResidualBlock)
        self.assertIsInstance(self.res_block_1.layers, torch.nn.Sequential)
        self.assertIsInstance(self.res_block_2.layers, torch.nn.Sequential)

    def test_forward(self) -> None:
        """
        Test the forward pass of the layer and mainly focus on the shape of the output
        """
        # Should work
        x_1 = torch.randn(1, 256, 5, 5)
        x_2 = torch.randn(1, 256, 100, 100)
        x_3 = torch.randn(1, 10, 5, 5)
        x_4 = torch.randn(1, 10, 100, 100)

        out_1 = self.res_block_1.forward(x_1)
        out_2 = self.res_block_1.forward(x_2)
        out_3 = self.res_block_2.forward(x_3)
        out_4 = self.res_block_2.forward(x_4)

        self.assertEqual(out_1.shape, x_1.shape)
        self.assertEqual(out_2.shape, x_2.shape)
        self.assertEqual(out_3.shape, x_3.shape)
        self.assertEqual(out_4.shape, x_4.shape)

        # Should raise an error
        x_3 = torch.randn(1, 200, 5, 5)
        x_4 = torch.randn(1, 256, 10)

        with self.assertRaises(RuntimeError):
            self.res_block_1.forward(x_3)
        with self.assertRaises(RuntimeError):
            self.res_block_2.forward(x_3)

        with self.assertRaises(ValueError):
            self.res_block_1.forward(x_4)
        with self.assertRaises(ValueError):
            self.res_block_2.forward(x_4)
