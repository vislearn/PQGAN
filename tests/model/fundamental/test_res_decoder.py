import os
import unittest

import torch

from vv.models.fundamental import ResDecoder

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestResDecoder(unittest.TestCase):
    """
    Test class for the Encoder class
    """
    def setUp(self) -> None:
        self.output_channels_rgb = 3
        self.output_channels_grey = 1
        self.input_units = 10
        self.batch_norm = True
        self.dropout_probability = 0.1

        self.model_imgNet = ResDecoder(output_size=(self.output_channels_rgb, 150, 150),
                                       input_units=self.input_units,
                                       dropout_prob=self.dropout_probability,
                                       batch_norm=self.batch_norm,
                                       verbose=False)
        self.model_cifar = ResDecoder(output_size=(self.output_channels_rgb, 32, 32),
                                      input_units=self.input_units,
                                      dropout_prob=self.dropout_probability,
                                      batch_norm=self.batch_norm,
                                      verbose=False)
        self.model_mnist = ResDecoder(output_size=(self.output_channels_grey, 28, 28),
                                      input_units=self.input_units,
                                      dropout_prob=self.dropout_probability,
                                      batch_norm=self.batch_norm,
                                      verbose=False)

        self.device = torch.device('cpu')  # specify the device as CPU

        self.z_imageNet = torch.randn(1, 10, 40, 40).to(self.device)
        self.z_cifar = torch.randn(1, 10, 11, 11).to(self.device)
        self.z_mnist = torch.randn(1, 10, 10, 10).to(self.device)

    def test_init(self) -> None:
        """
        Test the initialization of the Encoder class
        """
        # Check if the model is an instance of Encoder
        self.assertIsInstance(self.model_imgNet, ResDecoder)
        self.assertIsInstance(self.model_cifar, ResDecoder)
        self.assertIsInstance(self.model_mnist, ResDecoder)

    def test_forward(self) -> None:
        """
        Test the forward pass of the Encoder class
        """
        x_imageNet = self.model_imgNet(self.z_imageNet)
        x_cifar = self.model_cifar(self.z_cifar)
        x_mnist = self.model_mnist(self.z_mnist)

        # Check the shape of the output
        self.assertEqual(x_imageNet.shape[:2], torch.Size([1, self.output_channels_rgb]))
        self.assertEqual(x_cifar.shape[:2], torch.Size([1, self.output_channels_rgb]))
        self.assertEqual(x_mnist.shape[:2], torch.Size([1, self.output_channels_grey]))

        # Check the size of the last two dimensions
        output_size_imageNet = x_imageNet.size()[2:]
        output_size_cifar = x_cifar.size()[2:]
        output_size_mnist = x_mnist.size()[2:]

        self.assertEqual(output_size_imageNet[0], 150)
        self.assertEqual(output_size_imageNet[1], 150)
        self.assertEqual(output_size_cifar[0], 32)
        self.assertEqual(output_size_cifar[1], 32)
        self.assertEqual(output_size_mnist[0], 28)
        self.assertEqual(output_size_mnist[1], 28)

    def test_get_intermediate_sizes(self) -> None:
        """
        Test the get_intermediate_sizes method of the Encoder class
        """
        # Test get_intermediate_sizes
        intermediate_sizes_imageNet = self.model_imgNet.get_intermediate_sizes(self.z_imageNet)
        intermediate_sizes_cifar = self.model_cifar.get_intermediate_sizes(self.z_cifar)
        intermediate_sizes_mnist = self.model_mnist.get_intermediate_sizes(self.z_mnist)

        num_layers = 12

        self.assertEqual(len(intermediate_sizes_imageNet), num_layers)
        self.assertEqual(len(intermediate_sizes_cifar), num_layers)
        self.assertEqual(len(intermediate_sizes_mnist), num_layers)
