import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from counterfactual_xai.utils.mimic_dataloader import MimiDataLoader

torch.manual_seed(42)

df_clean = pd.read_csv("/Users/lukasscholz/repositorys/studienprojekt/counterfactuals/data/mimic/los_prediction.csv")

LOS = df_clean['LOS'].values
# Prediction Features
features = df_clean.drop(columns=['LOS'])

x_train, x_test, y_train, y_test = train_test_split(features,
                                                    LOS,
                                                    test_size=.20,
                                                    random_state=0)

class MimicDataset(Dataset):
    def __init__(self, X, y, scale_data=True, x_means=None, x_stds=None, y_means=None, y_stds=None):
        if scale_data:
            if x_means is not None and x_stds is not None:
                X = (X - x_means) / x_stds
            if y_means is not None and y_stds is not None:
                y = (y - y_means) / y_stds
        # X = X.to_numpy()
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(58, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def load_model_and_predict(x_new):
    # Define the model structure
    mlp = MLP()

    # Load the model parameters
    mlp.load_state_dict(torch.load('mlp_model.pth'))
    mlp.eval()

    # Scale the input
    # x_new_scaled = (x_new - x_means) / x_stds

    # Convert to tensor
    # x_new_tensor = x_new_scaled

    # Make predictions
    with torch.no_grad():
        y_pred = mlp(x_new)

    # Scale the predictions back
    # y_pred_rescaled = y_pred * y_stds + y_means

    return y_pred


def calculate_mse_rmse(x_test, y_test):
    y_pred = load_model_and_predict(x_test).numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mse, rmse


if __name__ == "__main__":
    # Create Dataset instances
    train_dataset = MimicDataset(x_train, y_train, scale_data=False)
    test_dataset = MimicDataset(x_test, y_test, scale_data=False)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    mlp = MLP()

    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # clip_value = 0.5
    loss_curve = []
    for epoch in range(0, 100):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_loader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # nn.utils.clip_grad_value_(mlp.parameters(), clip_value)

            # Perform optimization
            optimizer.step()

            # Print statistics

            current_loss += loss.item()
            if i % 10 == 0:
                loss_curve.append(current_loss / 500)
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

        # Process is complete.
    plt.plot(loss_curve)
    plt.savefig("loss_curve.png")

    torch.save(mlp.state_dict(), './mlp_model.pth')

    # print(load_model_and_predict(torch.tensor(x_test.to_numpy().astype(np.float32))))
    mse, rmse = calculate_mse_rmse(torch.tensor(x_test.astype(np.float32)), y_test)
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
