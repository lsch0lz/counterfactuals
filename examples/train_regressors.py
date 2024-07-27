import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

models = [SGDRegressor(random_state=0),
          GradientBoostingRegressor(random_state=0),
          LinearRegression(),
          KNeighborsRegressor(),
          RandomForestRegressor(random_state=0)]

df_clean = pd.read_csv("/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/los_prediction.csv")

LOS = df_clean['LOS'].values
# Prediction Features
features = df_clean.drop(columns=['LOS'])

CSV_PATH = "./../data/"
INPUT_DIMS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 41]
X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds, DATA_KEYS, input_dims = MimiDataLoader(INPUT_DIMS,
                                                                                                           CSV_PATH).get_mimic_dataset()

results = {}

for model in models:
    # Instantiate and fit Regressor Model
    reg_model = model
    reg_model.fit(X_train, np.ravel(y_train))

    # Make predictions with model
    y_test_preds = reg_model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_preds)
    rmse = np.sqrt(mse)
    print(f"MODEL: {str(model)}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print("---" * 10)

