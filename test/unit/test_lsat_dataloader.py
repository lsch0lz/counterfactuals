import os.path
import unittest
from pathlib import Path

from counterfactual_xai.utils.lsat_dataloader import LsatDataloader


class TestLsatDataloader(unittest.TestCase):
    DATA_DIR = str(Path(__file__).parents[1].resolve()) + "/data/"
    INPUT_DIM_VEC = [1, 1, 8, 2]

    def test_if_data_has_the_right_shape_after_trainsformation(self):
        data_loader = LsatDataloader(self.INPUT_DIM_VEC, self.DATA_DIR)
        x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = data_loader.get_lsat_dataset()

        assert x_train.shape == (17432, 12)
        assert x_test.shape == (4358, 12)

        assert y_train.shape == (17432, 1)
        assert y_test.shape == (4358, 1)
