import os.path
import unittest

from counterfactual_xai.utils.lsat_dataloader import LsatDataloader


class TestLsatDataloader(unittest.TestCase):
    DATA_DIR = "./../../data/"

    def test_if_test_and_train_file_is_downloaded(self):
        assert os.path.isfile(self.DATA_DIR + "law_school_cf_test.csv") == False
        assert os.path.isfile(self.DATA_DIR + "law_school_cf_test.csv") == False

        data_loader = LsatDataloader(self.DATA_DIR)
        data_loader.get_lsat_dataset()

        assert os.path.isfile(self.DATA_DIR + "law_school_cf_test.csv") == True
        assert os.path.isfile(self.DATA_DIR + "law_school_cf_train.csv") == True

        os.remove(self.DATA_DIR + "law_school_cf_test.csv")
        os.remove(self.DATA_DIR + "law_school_cf_train.csv")
