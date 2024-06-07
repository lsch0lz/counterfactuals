import numpy as np
import pandas as pd
from pandas import DataFrame


class MimiDataLoader:
    MIMIC_TEST_FILE = "test_mort_icu.csv"
    MIMIC_TRAIN_FILE = "train_mort_icu.csv"

    TARGET_KEY = "mort_icu"
    DATA_KEYS = ["age", "max_hours", "vent", "vaso", "adenosine", "dobutamine", "dopamine", "epinephrine", "isuprel",
                 "milrinone", "norepinephrine", "phenylephrine", "vasopressin", "colloid_bolus", "crystalloid_bolus", "nivdurations", "gender_F",
                 "gender_M", "ethnicity_AMERICAN INDIAN/ALASKA NATIVE", "ethnicity_AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE",
                 "ethnicity_ASIAN", "ethnicity_ASIAN - ASIAN INDIAN", "ethnicity_ASIAN - CAMBODIAN", "ethnicity_ASIAN - CHINESE",
                 "ethnicity_ASIAN - FILIPINO", "ethnicity_ASIAN - JAPANESE", "ethnicity_ASIAN - KOREAN", "ethnicity_ASIAN - OTHER",
                 "ethnicity_ASIAN - THAI", "ethnicity_ASIAN - VIETNAMESE", "ethnicity_BLACK/AFRICAN", "ethnicity_BLACK/AFRICAN AMERICAN",
                 "ethnicity_BLACK/CAPE VERDEAN", "ethnicity_BLACK/HAITIAN", "ethnicity_CARIBBEAN ISLAND", "ethnicity_HISPANIC OR LATINO",
                 "ethnicity_HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)", "ethnicity_HISPANIC/LATINO - COLOMBIAN", "ethnicity_HISPANIC/LATINO - CUBAN",
                 "ethnicity_HISPANIC/LATINO - DOMINICAN", "ethnicity_HISPANIC/LATINO - GUATEMALAN", "ethnicity_HISPANIC/LATINO - HONDURAN",
                 "ethnicity_HISPANIC/LATINO - MEXICAN", "ethnicity_HISPANIC/LATINO - PUERTO RICAN", "ethnicity_HISPANIC/LATINO - SALVADORAN",
                 "ethnicity_MIDDLE EASTERN", "ethnicity_MULTI RACE ETHNICITY", "ethnicity_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
                 "ethnicity_OTHER", "ethnicity_PATIENT DECLINED TO ANSWER", "ethnicity_PORTUGUESE", "ethnicity_SOUTH AMERICAN",
                 "ethnicity_UNABLE TO OBTAIN", "ethnicity_UNKNOWN/NOT SPECIFIED", "ethnicity_WHITE", "ethnicity_WHITE - BRAZILIAN",
                 "ethnicity_WHITE - EASTERN EUROPEAN", "ethnicity_WHITE - OTHER EUROPEAN", "ethnicity_WHITE - RUSSIAN"]

    def __init__(self, input_dims, csv_path: str = None):
        self.input_dims = input_dims
        self.csv_path = csv_path

    def _input_dim_vec_to_X_dims(self):
        """This is for our cat_Gauss VAE model"""
        return np.concatenate([np.full(shape=i, fill_value=i) for i in self.input_dims])

    def _X_dims_to_input_dim_vec(self, X_dims):
        """This is for our cat_Gauss VAE model"""
        input_dim_vec = []
        i = 0
        while i < len(X_dims):
            input_dim_vec.append(X_dims[i])
            i += X_dims[i]
        return np.array(input_dim_vec)
    def get_mimic_dataset(self):
        df_train: DataFrame = pd.read_csv(self.csv_path + self.MIMIC_TRAIN_FILE)
        data_train = df_train.to_dict("list")
        train_keys = list(data_train.keys())

        df_test: DataFrame = pd.read_csv(self.csv_path + self.MIMIC_TEST_FILE)
        data_test = df_test.to_dict("list")

        X_dimensions = self._input_dim_vec_to_X_dims()
        input_dims = self._X_dims_to_input_dim_vec(X_dimensions)

        X_train = np.empty((len(data_train[train_keys[0]]), len(self.DATA_KEYS)))
        X_test = np.empty((len(data_test[train_keys[0]]), len(self.DATA_KEYS)))

        y_train = np.array(data_train[self.TARGET_KEY]).reshape(-1, 1)
        y_test = np.array(data_test[self.TARGET_KEY]).reshape(-1, 1)

        for k_idx, k in enumerate(self.DATA_KEYS):
            X_train[:, k_idx] = np.array(data_train[k])
            X_test[:, k_idx] = np.array(data_test[k])

        x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)

        """
        # Adjust x_means and x_stds for specific dimensions
        x_means_adjusted = np.copy(x_means)
        x_stds_adjusted = np.copy(x_stds)
        for idx in range(len(X_dimensions)):
            if X_dimensions[idx] > 1:
                x_means_adjusted[idx] = 0
                x_stds_adjusted[idx] = 1

        x_stds_adjusted[x_stds_adjusted < 1e-10] = 1

        x_train = ((X_train - x_means_adjusted) / x_stds_adjusted).astype(np.float32)
        x_test = ((X_test - x_means_adjusted) / x_stds_adjusted).astype(np.float32)
        """

        y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)
        y_stds[y_stds < 1e-10] = 1

        # y_train = ((y_train - y_means) / y_stds).astype(np.float32)
        # y_test = ((y_test - y_means) / y_stds).astype(np.float32)

        return X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds, self.DATA_KEYS, input_dims

