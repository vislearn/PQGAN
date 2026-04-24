import os
import unittest

import pytest
import torch
from torch.utils.data import DataLoader

import vv.data as data_loader

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No cuda on Github.")


class TestMNISTLoader(unittest.TestCase):
    """
    Test class for the Decoder class
    """
    def setUp(self) -> None:
        """
        Set up the test. Gets called as first function
        """
        self.train_val_split = 0.8
        self.batch_size = 32
        self.workers = 4
        self.verbose = False

        self.mnist_loader = data_loader.MNISTLoader(train_val_split=self.train_val_split,
                                                    batch_size=self.batch_size,
                                                    workers=self.workers,
                                                    verbose=self.verbose)

    def test_init(self) -> None:
        """
        Test the initialization of the MNIStLoader class
        """
        self.assertIsInstance(self.mnist_loader, data_loader.MNISTLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_prepare_data(self) -> None:
        """
        Test the prepare_data method
        """
        self.mnist_loader.prepare_data()
        self.assertTrue(os.path.exists(os.path.join(self.mnist_loader.data_dir,
                                                    'MNIST')))

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_training(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.mnist_loader.setup('fit')

        self.assertIsNotNone(self.mnist_loader.dataset_train)
        self.assertIsNotNone(self.mnist_loader.dataset_val)
        self.assertTrue(len(self.mnist_loader.dataset_train) > len(self.mnist_loader.dataset_val))
        self.assertTrue(self.mnist_loader.dataset_train[0][0].shape == torch.Size([1, 28, 28]))
        self.assertTrue(self.mnist_loader.dataset_val[0][0].shape == torch.Size([1, 28, 28]))
        self.assertTrue(type(self.mnist_loader.dataset_train[0][1]) is int)
        self.assertTrue(type(self.mnist_loader.dataset_val[0][1]) is int)

        # Data loaders
        train_loader = self.mnist_loader.train_dataloader()
        val_loader = self.mnist_loader.val_dataloader()

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_testing(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.mnist_loader.setup("test")

        self.assertIsNotNone(self.mnist_loader.dataset_test)
        self.assertTrue(len(self.mnist_loader.dataset_test) > 0)
        self.assertTrue(self.mnist_loader.dataset_test[0][0].shape == torch.Size([1, 28, 28]))
        self.assertTrue(type(self.mnist_loader.dataset_test[0][1]) is int)

        # Data loaders
        test_loader = self.mnist_loader.test_dataloader()

        self.assertIsInstance(test_loader, DataLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_predicting(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.mnist_loader.setup("predict")

        self.assertIsNotNone(self.mnist_loader.dataset_predict)
        self.assertTrue(len(self.mnist_loader.dataset_predict) > 0)
        self.assertTrue(self.mnist_loader.dataset_predict[0][0].shape == torch.Size([1, 28, 28]))
        self.assertTrue(type(self.mnist_loader.dataset_predict[0][1]) is int)

        # Data loaders
        predict_loader = self.mnist_loader.predict_dataloader()

        self.assertIsInstance(predict_loader, DataLoader)
