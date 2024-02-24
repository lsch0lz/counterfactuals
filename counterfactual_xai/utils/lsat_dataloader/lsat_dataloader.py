import os.path
import logging

import requests
import csv

import numpy as np
import pandas as pd
from pandas import DataFrame


class LsatDataloader:
    LSAT_TEST_FILE = "law_school_cf_test.csv"
    LSAT_TRAIN_FILE = "law_school_cf_train.csv"
    DOWNLOAD_ADRESS = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/"

    TARGET_KEY = 'ZFYA'
    DATA_KEYS = ['LSAT', 'UGPA', 'amerind', 'mexican', 'other', 'black', 'asian', 'puerto', 'hisp', 'white', 'female', 'male']

    def __init__(self, input_dims, csv_path: str = None):
        self.input_dims = input_dims
        self.csv_path = csv_path

    def input_dim_vec_to_X_dims(self):
        """This is for our cat_Gauss VAE model"""
        return np.concatenate([np.full(shape=i, fill_value=i) for i in self.input_dims])

    def _order_data_entries(self):
        pass

    def _download_lsat_from_github(self, file: str):
        downloadfile = self.LSAT_TEST_FILE if file == "test" else self.LSAT_TRAIN_FILE
        with requests.Session() as s:
            download = s.get(self.DOWNLOAD_ADRESS + downloadfile)
            decoded_content = download.content.decode('utf-8')
            csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')

            with open(str(self.csv_path + downloadfile), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in csv_reader:
                    writer.writerow(row)

    def _check_for_local_lsat_files(self):
        if not os.path.isfile(self.csv_path + self.LSAT_TEST_FILE):
            self._download_lsat_from_github(file="test")

        if not os.path.isfile(self.csv_path + self.LSAT_TRAIN_FILE):
            self._download_lsat_from_github(file="train")

    def get_lsat_dataset(self):
        self._check_for_local_lsat_files()

        df_train: DataFrame = pd.read_csv(self.csv_path + self.LSAT_TRAIN_FILE)
        data_train = df_train.to_dict("list")
        train_keys = list(data_train.keys())
        logging.info("Train Keys: %s", train_keys)

        df_test: DataFrame = pd.read_csv(self.csv_path + self.LSAT_TEST_FILE)
        data_test = df_test.to_dict("list")

        X_dimensions = self.input_dim_vec_to_X_dims()

        X_train = np.empty((len(data_train[train_keys[0]]), len(self.DATA_KEYS)))
        X_test = np.empty((len(data_test[train_keys[0]]), len(self.DATA_KEYS)))

        y_train = np.array(data_train[self.TARGET_KEY]).reshape(-1, 1)
        y_test = np.array(data_test[self.TARGET_KEY]).reshape(-1, 1)

        for k_idx, k in enumerate(self.DATA_KEYS):
            X_train[:, k_idx] = np.array(data_train[k])
            X_test[:, k_idx] = np.array(data_test[k])

        x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
        x_means[X_dimensions > 1] = 0
        x_stds[X_dimensions > 1] = 1
        x_stds[x_stds < 1e-10] = 1

        x_train = ((X_train - x_means) / x_stds).astype(np.float32)
        x_test = ((X_test - x_means) / x_stds).astype(np.float32)

        y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)
        y_stds[y_stds < 1e-10] = 1

        y_train = ((y_train - y_means) / y_stds).astype(np.float32)
        y_test = ((y_test - y_means) / y_stds).astype(np.float32)

        return x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds, self.DATA_KEYS, self.input_dims
