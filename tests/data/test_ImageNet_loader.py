import os
import unittest

import pytest
import torch
from torch.utils.data import DataLoader

import vv.data as data_loader

# import pytest
# import torch
# from torch.utils.data import DataLoader


IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No cuda on Github.")


class TestImageNetLoader(unittest.TestCase):
    """
    Test class for the Decoder class
            """
    def setUp(self) -> None:
        """
        Set up the test. Gets called as first function
        """
        self.train_instances = 100
        self.val_instances = 10
        self.test_instances = 10
        self.batch_size = 5
        self.workers = 4
        self.verbose = False

        self.imageNet_loader = data_loader.ImageNetLoader(train_instances=self.train_instances,
                                                          val_instances=self.val_instances,
                                                          test_instances=self.test_instances,
                                                          batch_size=self.batch_size,
                                                          workers=self.workers,
                                                          verbose=self.verbose,
                                                          data_dir="/path/to/imageNet")

    def test_init(self) -> None:
        """
        Test the initialization of the ImageNetLoader
        """
        self.assertIsInstance(self.imageNet_loader, data_loader.ImageNetLoader)

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_prepare_data(self) -> None:
        """
        Test the prepare_data method of the ImageNetLoader
        """
        try:
            self.imageNet_loader.prepare_data()
        except Exception as e:
            self.fail(f"prepare_data raised an exception: {e}")

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_setup_fit(self) -> None:
        """
        Test the setup method for the 'fit' stage
        """
        self.imageNet_loader.setup('fit')

        self.assertIsNotNone(self.imageNet_loader.dataset_train)
        self.assertIsNotNone(self.imageNet_loader.dataset_val)
        self.assertTrue(len(self.imageNet_loader.dataset_train) > 0)
        self.assertTrue(len(self.imageNet_loader.dataset_val) > 0)

        for data in self.imageNet_loader.dataset_val:
            # print(data[0].shape)
            self.assertTrue(data[0].shape == torch.Size([3, 128, 128]))
            self.assertTrue(type(data[1]) is int)
            break

        for data in self.imageNet_loader.dataset_train:
            # print(data[0].shape)
            self.assertTrue(data[0].shape == torch.Size([3, 128, 128]))
            self.assertTrue(type(data[1]) is int)
            break

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_setup_test(self) -> None:
        """
        Test the setup method for the 'test' stage
        """
        self.imageNet_loader.setup('test')

        self.assertIsNotNone(self.imageNet_loader.dataset_test)
        self.assertTrue(len(self.imageNet_loader.dataset_test) > 0)

        for data in self.imageNet_loader.dataset_test:
            self.assertTrue(data[0].shape == torch.Size([3, 128, 128]))
            self.assertTrue(type(data[1]) is int)
            break

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_train_dataloader(self) -> None:
        """
        Test the train_dataloader method
        """
        self.imageNet_loader.setup('fit')
        train_loader = self.imageNet_loader.train_dataloader()

        self.assertIsInstance(train_loader, DataLoader)
        self.assertTrue(len(train_loader) > 0)

        # Check batch size
        self.assertTrue(train_loader.batch_size == self.batch_size)
        for data in train_loader:
            self.assertTrue(data[0].shape[0] == self.batch_size)
            break

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_val_dataloader(self) -> None:
        """
        Test the val_dataloader method
        """
        self.imageNet_loader.setup('fit')
        val_loader = self.imageNet_loader.val_dataloader()

        self.assertIsInstance(val_loader, DataLoader)
        self.assertTrue(len(val_loader) > 0)

        # Check batch size
        self.assertTrue(val_loader.batch_size == self.batch_size)
        for data in val_loader:
            self.assertTrue(data[0].shape[0] == self.batch_size)
            break

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_test_dataloader(self) -> None:
        """
        Test the test_dataloader method
        """
        self.imageNet_loader.setup('test')
        test_loader = self.imageNet_loader.test_dataloader()

        self.assertIsInstance(test_loader, DataLoader)
        self.assertTrue(len(test_loader) > 0)

        # Check batch size
        self.assertTrue(test_loader.batch_size == self.batch_size)
        for data in test_loader:
            self.assertTrue(data[0].shape[0] == self.batch_size)
            break

    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not download large data sets in GitHub Actions")
    def test_predict_dataloader(self) -> None:
        """
        Test the predict_dataloader method
        """
        self.imageNet_loader.setup('predict')
        predict_loader = self.imageNet_loader.predict_dataloader()

        self.assertIsInstance(predict_loader, DataLoader)
        self.assertTrue(len(predict_loader) > 0)

        # Check batch size
        self.assertTrue(predict_loader.batch_size == self.batch_size)
        for data in predict_loader:
            self.assertTrue(data[0].shape[0] == self.batch_size)
            break
