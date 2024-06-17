from torch import nn
import torch.nn.functional as F


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(GaussianMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 2 * output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        x = self.block(x)
        # mu = x[:, :self.output_dim]
        # sigma = F.softplus(x[:, self.output_dim:])
        mu = x[0]
        sigma = F.softplus(x[1])
        return mu, sigma
