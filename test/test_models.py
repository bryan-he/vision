from __future__ import division
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch._utils_internal import get_file_path_2
import unittest
import math
import random
import numpy as np

model_names = sorted(name for name in torchvision.models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.__dict__[name]))


class Tester(unittest.TestCase):
    # TODO: there should be tests for GPU and pretrained weights

    def test_forward_during_training(self):
        """Evaluate the model at training time, and check that the output has the right shape and no nans."""

        for name in model_names:
            model = torchvision.models.__dict__[name]()

            if name.startswith("inception"):
                window = [299, 512]
            else:
                window = [224, 299, 512]

            for w in window:
                x = torch.randn(2, 3, w, w)
                pred = model(x)

                if name.startswith("inception"):
                    self.assertEqual(len(pred), 2)
                    self.assertEqual(pred[0].shape, torch.Size([2, 1000]))
                    self.assertEqual(pred[1].shape, torch.Size([2, 1000]))
                    self.assertFalse(torch.isnan(pred[0]).any())
                    self.assertFalse(torch.isnan(pred[1]).any())
                else:
                    self.assertEqual(pred.shape, torch.Size([2, 1000]))
                    self.assertFalse(torch.isnan(pred).any())

    def test_forward_during_testing(self):
        """Evaluate the model at testing time, and check that the output has the right shape and no nans.
           Additionally verifies that running forward on the same input gives a consistent prediction
           (for things like dropout and batchnorm).
        """

        for name in model_names:
            model = torchvision.models.__dict__[name]()
            model.train(False)

            if name.startswith("inception"):
                window = [299, 512]
            else:
                window = [224, 299, 512]

            for w in window:
                for channels in [1, 2]:
                    x = torch.randn(channels, 3, w, w)
                    pred = model(x)
                    self.assertEqual(pred.shape, torch.Size([channels, 1000]))
                    self.assertFalse(torch.isnan(pred).any())

                    pred2 = model(x)
                    self.assertEqual(pred.shape, torch.Size([channels, 1000]))
                    self.assertFalse(torch.isnan(pred2).any())

                    self.assertTrue((pred == pred2).all())


if __name__ == '__main__':
    unittest.main()
