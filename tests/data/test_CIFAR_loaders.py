import os
import unittest

import pytest
from torch.utils.data import DataLoader

import vv.data as data_loader

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No cuda on Github.")


class TestCIFARLoaders(unittest.TestCase):
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

        self.cifar10_loader = data_loader.CIFAR10Loader(train_val_split=self.train_val_split,
                                                        batch_size=self.batch_size,
                                                        workers=self.workers,
                                                        verbose=self.verbose)
        self.cifar100_loader = data_loader.CIFAR100Loader(train_val_split=self.train_val_split,
                                                          batch_size=self.batch_size,
                                                          workers=self.workers,
                                                          verbose=self.verbose)

    def test_init(self) -> None:
        """
        Test the initialization of the MNIStLoader class
        """
        self.assertIsInstance(self.cifar10_loader, data_loader.CIFAR10Loader)
        self.assertIsInstance(self.cifar100_loader, data_loader.CIFAR100Loader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_prepare_data(self) -> None:
        """
        Test the prepare_data method
        """
        self.cifar10_loader.prepare_data()
        self.cifar100_loader.prepare_data()
        self.assertTrue(os.path.exists(os.path.join(self.cifar10_loader.data_dir,
                                                    'cifar-100-python.tar.gz')))
        self.assertTrue(os.path.exists(os.path.join(self.cifar100_loader.data_dir,
                                                    'cifar-10-python.tar.gz')))

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_training(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.cifar10_loader.setup('fit')
        self.cifar100_loader.setup('fit')

        self.assertIsNotNone(self.cifar10_loader.dataset_train)
        self.assertIsNotNone(self.cifar100_loader.dataset_train)
        self.assertIsNotNone(self.cifar10_loader.dataset_val)
        self.assertIsNotNone(self.cifar100_loader.dataset_val)
        self.assertTrue(len(self.cifar10_loader.dataset_train) > len(self.cifar10_loader.dataset_val))
        self.assertTrue(len(self.cifar100_loader.dataset_train) > len(self.cifar100_loader.dataset_val))
        self.assertTrue(self.cifar10_loader.dataset_train[0][0].shape == (3, 32, 32))
        self.assertTrue(self.cifar100_loader.dataset_train[0][0].shape == (3, 32, 32))
        self.assertTrue(self.cifar10_loader.dataset_val[0][0].shape == (3, 32, 32))
        self.assertTrue(self.cifar100_loader.dataset_val[0][0].shape == (3, 32, 32))
        self.assertTrue(type(self.cifar10_loader.dataset_train[0][1]) is int)
        self.assertTrue(type(self.cifar100_loader.dataset_train[0][1]) is int)
        self.assertTrue(type(self.cifar10_loader.dataset_val[0][1]) is int)
        self.assertTrue(type(self.cifar100_loader.dataset_val[0][1]) is int)

        # Data loaders
        train_loader10 = self.cifar10_loader.train_dataloader()
        val_loader10 = self.cifar10_loader.val_dataloader()
        train_loader100 = self.cifar100_loader.train_dataloader()
        val_loader100 = self.cifar100_loader.val_dataloader()

        self.assertIsInstance(train_loader10, DataLoader)
        self.assertIsInstance(val_loader10, DataLoader)
        self.assertIsInstance(train_loader100, DataLoader)
        self.assertIsInstance(val_loader100, DataLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_testing(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.cifar10_loader.setup("test")
        self.cifar100_loader.setup("test")

        self.assertIsNotNone(self.cifar10_loader.dataset_test)
        self.assertIsNotNone(self.cifar100_loader.dataset_test)
        self.assertTrue(len(self.cifar10_loader.dataset_test) > 0)
        self.assertTrue(len(self.cifar100_loader.dataset_test) > 0)
        self.assertTrue(self.cifar10_loader.dataset_test[0][0].shape == (3, 32, 32))
        self.assertTrue(self.cifar100_loader.dataset_test[0][0].shape == (3, 32, 32))
        self.assertTrue(type(self.cifar10_loader.dataset_test[0][1]) is int)
        self.assertTrue(type(self.cifar100_loader.dataset_test[0][1]) is int)

        # Data loaders
        test_loader10 = self.cifar10_loader.test_dataloader()
        test_loader100 = self.cifar100_loader.test_dataloader()

        self.assertIsInstance(test_loader10, DataLoader)
        self.assertIsInstance(test_loader100, DataLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_predicting(self) -> None:
        """
        Test the setup method the data loaders
        """

        # Setup
        self.cifar10_loader.setup("predict")
        self.cifar100_loader.setup("predict")

        self.assertIsNotNone(self.cifar10_loader.dataset_predict)
        self.assertIsNotNone(self.cifar100_loader.dataset_predict)
        self.assertTrue(len(self.cifar10_loader.dataset_predict) > 0)
        self.assertTrue(len(self.cifar100_loader.dataset_predict) > 0)
        self.assertTrue(self.cifar10_loader.dataset_predict[0][0].shape == (3, 32, 32))
        self.assertTrue(self.cifar100_loader.dataset_predict[0][0].shape == (3, 32, 32))
        self.assertTrue(type(self.cifar10_loader.dataset_predict[0][1]) is int)
        self.assertTrue(type(self.cifar100_loader.dataset_predict[0][1]) is int)

        # Data loaders
        predict_loader_10 = self.cifar10_loader.predict_dataloader()
        predict_loader_100 = self.cifar100_loader.predict_dataloader()

        self.assertIsInstance(predict_loader_10, DataLoader)
        self.assertIsInstance(predict_loader_100, DataLoader)
