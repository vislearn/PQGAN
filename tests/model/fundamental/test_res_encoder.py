import os
import unittest

import pytest
import torch

from vv.models.fundamental import ResEncoder

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Nvidia CUDA is not available in GitHub Actions")
class TestResEncoder(unittest.TestCase):
    """
    Test class for the Encoder class
    """
    def setUp(self) -> None:
        self.input_channels_rgb = 3
        self.input_channels_grey = 1
        self.output_units = 10
        self.batch_norm = True
        self.dropout_probability = 0.1
        self.heads = 4

        self.model_imgNet = ResEncoder(input_size=[self.input_channels_rgb, 150, 150],
                                       output_units=self.output_units,
                                       dropout_prob=self.dropout_probability,
                                       batch_norm=self.batch_norm,
                                       verbose=False)

        self.model_cifar = ResEncoder(input_size=[self.input_channels_rgb, 32, 32],
                                      output_units=self.output_units,
                                      dropout_prob=self.dropout_probability,
                                      batch_norm=self.batch_norm,
                                      verbose=False)

        self.model_mnist = ResEncoder(input_size=[self.input_channels_grey, 28, 28],
                                      output_units=self.output_units,
                                      dropout_prob=self.dropout_probability,
                                      batch_norm=self.batch_norm,
                                      verbose=False)

        self.device = torch.device('cpu')  # specify the device as CPU

        self.x_imageNet = torch.randn(1, 3, 150, 150).to(self.device)
        self.x_cifar = torch.randn(1, 3, 32, 32).to(self.device)
        self.x_mnist = torch.randn(1, 1, 28, 28).to(self.device)

    def test_init(self) -> None:
        """
        Test the initialization of the Encoder class
        """
        # Check if the model is an instance of Encoder
        self.assertIsInstance(self.model_imgNet, ResEncoder)
        self.assertIsInstance(self.model_cifar, ResEncoder)
        self.assertIsInstance(self.model_mnist, ResEncoder)

    def test_forward(self) -> None:
        """
        Test the forward pass of the Encoder class
        """
        z_imageNet = self.model_imgNet(self.x_imageNet)
        z_cifar = self.model_cifar(self.x_cifar)
        z_mnist = self.model_mnist(self.x_mnist)

        # Check the shape of the output
        self.assertEqual(z_imageNet.shape, torch.Size([1, self.output_units, 40, 40]))
        self.assertEqual(z_cifar.shape, torch.Size([1, self.output_units, 11, 11]))
        self.assertEqual(z_mnist.shape, torch.Size([1, self.output_units, 10, 10]))

    def test_get_intermediate_sizes(self) -> None:
        """
        Test the get_intermediate_sizes method of the Encoder class
        """
        # Test get_intermediate_sizes
        intermediate_sizes_imageNet = self.model_imgNet.get_intermediate_sizes(self.x_imageNet)
        intermediate_sizes_cifar = self.model_cifar.get_intermediate_sizes(self.x_cifar)
        intermediate_sizes_mnist = self.model_mnist.get_intermediate_sizes(self.x_mnist)

        layer_number = 12

        self.assertEqual(len(intermediate_sizes_imageNet), layer_number)
        self.assertEqual(len(intermediate_sizes_cifar), layer_number)
        self.assertEqual(len(intermediate_sizes_mnist), layer_number)

    def test_get_additional_loss_objective(self) -> None:
        """
        Test the get_additional_loss_objective method of the Encoder class
        """

        # Get the additional loss objective
        loss_1 = self.model_imgNet.get_additional_loss_objective()

        # Check the type of the output
        self.assertIsInstance(loss_1, torch.Tensor)

        # Check the shape of the output
        self.assertEqual(loss_1.shape, torch.Size([]))

        # Check the value of the output
        self.assertEqual(loss_1.item(), 0.0)
