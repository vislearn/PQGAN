import os
import unittest

import torch

from vv.special_layers import VariationalLayer

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestVariationalLayer(unittest.TestCase):
    """
    Test class for the variational layer
    """
    def setUp(self) -> None:
        """
        Set up the test
        """
        self.input_dim = 10
        self.output_dim = 8
        self.vq = VariationalLayer(input_dim=self.input_dim,
                                   output_dim=self.output_dim)

    def test_init(self) -> None:
        """
        Test the initialization of the autoencoder class
        """
        self.assertIsInstance(self.vq, VariationalLayer)
        self.assertEqual(self.vq.input_dim, self.input_dim)
        self.assertEqual(self.vq.output_dim, self.output_dim)
        self.assertEqual(self.vq.mu_layer.weight.size(), torch.Size([self.output_dim, self.input_dim]))
        self.assertEqual(self.vq.logvar_layer.weight.size(), torch.Size([self.output_dim, self.input_dim]))

    def test_forward(self) -> None:
        """
        Test the forward pass of the layer and mainly focus on the shape of the output
        """
        # Should work
        x_1 = torch.randn(5, self.input_dim)
        x_2 = torch.randn(200, self.input_dim)

        z_1, mu_1, logvar_1 = self.vq.forward(x_1)
        z_2, mu_2, logvar_2 = self.vq.forward(x_2)

        self.assertEqual(z_1.size(), torch.Size([5, self.output_dim]))
        self.assertEqual(z_2.size(), torch.Size([200, self.output_dim]))
        self.assertEqual(mu_1.size(), torch.Size([5, self.output_dim]))
        self.assertEqual(mu_2.size(), torch.Size([200, self.output_dim]))
        self.assertEqual(logvar_1.size(), torch.Size([5, self.output_dim]))
        self.assertEqual(logvar_2.size(), torch.Size([200, self.output_dim]))

        # Should raise an error
        x_3 = torch.randn(3, self.input_dim + 1)
        x_5 = torch.randn(3, 2, 4)

        with self.assertRaises(RuntimeError):
            self.vq.forward(x_3)

        with self.assertRaises(RuntimeError):
            self.vq.forward(x_5)
