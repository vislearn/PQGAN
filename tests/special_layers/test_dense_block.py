import os
import unittest

import torch

from vv.special_layers import DenseBlock

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestDenseBlock(unittest.TestCase):
    """
    Test class for the dense block
    """
    def setUp(self) -> None:
        """
        Set up the test
        """

        self.dense_block_small = DenseBlock(input_channels=10,
                                            growth_rate=5,
                                            num_layers=3,
                                            reduce_channels=False,
                                            batch_norm=False,
                                            dropout_probability=0.0)
        self.dense_block_big = DenseBlock(input_channels=256,
                                          growth_rate=32,
                                          num_layers=5,
                                          reduce_channels=True,
                                          batch_norm=True,
                                          dropout_probability=0.1)

    def test_init(self) -> None:
        """
        Test the initialization of the autoencoder class
        """
        self.assertIsInstance(self.dense_block_small.layers, torch.nn.ModuleList)
        self.assertIsInstance(self.dense_block_big.layers, torch.nn.ModuleList)
        self.assertEqual(len(self.dense_block_small.layers), 3)
        self.assertEqual(len(self.dense_block_big.layers), 6)

        # Count the number of layers
        count_small = 0
        count_big = 0
        for layer in self.dense_block_small.layers:
            count_small += len(layer)
        for layer in self.dense_block_big.layers:
            count_big += len(layer)
        self.assertEqual(count_small, 6)
        self.assertEqual(count_big, 24)

    def test_forward(self) -> None:
        """
        Test the forward pass of the layer and mainly focus on the shape of the output
        """
        # Should work
        x_1_size_small = torch.Size([10, 10, 1, 1])
        x_2_size_small = torch.Size([10, 10, 32, 32])
        x_3_size_small = torch.Size([10, 10, 128, 128])
        x_1_size_big = torch.Size([10, 256, 1, 1])
        x_2_size_big = torch.Size([10, 256, 32, 32])
        x_3_size_big = torch.Size([10, 256, 128, 128])

        x_1_small = torch.randn(*x_1_size_small)
        x_2_small = torch.randn(*x_2_size_small)
        x_3_small = torch.randn(*x_3_size_small)
        x_1_big = torch.randn(*x_1_size_big)
        x_2_big = torch.randn(*x_2_size_big)
        x_3_big = torch.randn(*x_3_size_big)

        out_1_small = self.dense_block_small.forward(x_1_small)
        out_2_small = self.dense_block_small.forward(x_2_small)
        out_3_small = self.dense_block_small.forward(x_3_small)

        out_1_big = self.dense_block_big.forward(x_1_big)
        out_2_big = self.dense_block_big.forward(x_2_big)
        out_3_big = self.dense_block_big.forward(x_3_big)

        self.assertEqual(out_1_small.shape, torch.Size([10, 25, 1, 1]))
        self.assertEqual(out_2_small.shape, torch.Size([10, 25, 32, 32]))
        self.assertEqual(out_3_small.shape, torch.Size([10, 25, 128, 128]))

        self.assertEqual(out_1_big.shape, x_1_size_big)
        self.assertEqual(out_2_big.shape, x_2_size_big)
        self.assertEqual(out_3_big.shape, x_3_size_big)

        # Should raise an error
        x_4 = torch.randn(1, 10, 5, 5, 5)
        x_5 = torch.randn(1, 256, 10)

        with self.assertRaises(RuntimeError):
            self.dense_block_small.forward(x_4)
        with self.assertRaises(RuntimeError):
            self.dense_block_small.forward(x_5)

        with self.assertRaises(ValueError):
            self.dense_block_big.forward(x_4)
        with self.assertRaises(ValueError):
            self.dense_block_big.forward(x_5)
