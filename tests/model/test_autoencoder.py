import os
import unittest

import pytest
import torch
import torch.nn as nn

from vv.models import Autoencoder
from vv.models.fundamental import ResDecoder, ResEncoder, VQEncoder
from vv.utilities import delete_file, get_dir

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestAutoencoder(unittest.TestCase):
    """
    Test class for the autoencoder class
    """
    def setUp(self) -> None:
        """
        Set up the test
        """
        self.data_size = [3, 32, 32]
        self.out_shape = [10, 11, 11]
        self.codebook_splits = [2, 11, 11]
        self.separate_codebooks = [True, False, False]
        self.dropout_prob = 0.1
        self.batch_norm = True
        self.units = 10

        self.conv_base_encoder = ResEncoder(input_size=self.data_size,
                                            output_units=self.units,
                                            dropout_prob=self.dropout_prob,
                                            batch_norm=self.batch_norm,
                                            verbose=False)
        self.conv_base_decoder = ResDecoder(output_size=self.data_size,
                                            input_units=self.units,
                                            dropout_prob=self.dropout_prob,
                                            batch_norm=self.batch_norm,
                                            verbose=False)

        self.encoder = VQEncoder(base_encoder=self.conv_base_encoder,
                                 codebook_splits=self.codebook_splits,
                                 separate_codebooks=self.separate_codebooks,
                                 codebook_size=self.units,
                                 decoder_encoder_ste=True,
                                 use_commitment_loss=True,
                                 vq_loss="mse",
                                 embedding_loss_weight=0.25,
                                 verbose=False)

        self.model = Autoencoder(encoder=self.encoder,
                                 decoder=self.conv_base_decoder,
                                 disconnect_decoder_loss_propagation_from_encoder=False,
                                 latent_error_weight=1.0,
                                 learning_rate=1e-3,
                                 lr_patience=1)

    def test_init(self) -> None:
        """
        Test the initialization of the autoencoder class
        """

        self.assertTrue(isinstance(self.model, Autoencoder))
        self.assertTrue(isinstance(self.model, nn.Module))

    def test_encode(self) -> None:
        """
        Test the encode method of the autoencoder class
        """

        # Generate random input data
        x_1 = torch.randn(32, *self.data_size)
        x_2 = torch.randn(32, 5, 5)

        # Pass the input data through the model
        output_1 = self.model.encode(x_1)

        with pytest.raises(RuntimeError):
            _ = self.model.encode(x_2)

        # Check the shape of the output
        self.assertEqual(output_1.shape, (32, *self.out_shape))

    def test_decode(self) -> None:
        """
        Test the decode method of the autoencoder class
        """

        # Generate random input data
        x_1 = torch.randn(32, *self.out_shape)

        # Pass the input data through the model
        output_1 = self.model.decode(x_1)

        # Check the shape of the output
        self.assertEqual(output_1.shape, (32, *self.data_size))

    def test_forward(self) -> None:
        """
        Test the forward pass of the LinearDecoder class
        """

        # Generate random input data
        x_1 = torch.randn(32, 3, 32, 32)

        # Pass the input data through the model
        output_1 = self.model(x_1)

        # Check the shape of the output
        self.assertEqual(output_1.shape, x_1.shape)

    def test_sample(self) -> None:
        """
        Test the sample method of the autoencoder class
        """

        # Generate random input data
        n_samples = 32

        # Pass the input data through the model
        output = self.model.sample(n_samples)

        # Check the shape of the output
        self.assertEqual(output.shape, (n_samples, 3, 32, 32))

    def test_steps(self) -> None:
        """
        Test the training step of the autoencoder class
        """

        # Generate random input data
        x = (torch.randn(32, 3, 32, 32), 0)

        # Pass the input data through the model
        output_train = self.model.training_step(x)
        output_val = self.model.validation_step(x, 1)
        output_test = self.model.test_step(x)

        # Check the type of the output
        self.assertIsInstance(output_train, torch.Tensor)
        self.assertEqual(output_train.shape, ())
        self.assertEqual(output_val, None)
        self.assertEqual(output_test, None)

    def test_configure_optimizers(self) -> None:
        """
        Test the configure_optimizers method of the autoencoder class
        """

        # Pass the input data through the model
        optimizer = self.model.configure_optimizers()

        # Check the type of the output
        self.assertTrue(isinstance(optimizer, dict))
        self.assertIsInstance(optimizer["optimizer"], torch.optim.Optimizer)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large models in GitHub Actions")
    def test_visualize_gradient_flow(self) -> None:
        """
        Test the visualize_gradient_flow method of the autoencoder class
        """

        # Generate random input data
        x = torch.randn(32, *self.data_size)

        dir = get_dir("__tmp__")
        name = "gradient_flow"
        full_dir = os.path.join(dir, name)

        # Pass the input data through the model
        self.model.visualize_gradient_flow(x=x,
                                           filepath=full_dir,
                                           show_gradients=True)

        # Force delete directory
        delete_file(full_dir)
        delete_file(os.path.join(dir, name + ".png"))
        os.rmdir(dir)
