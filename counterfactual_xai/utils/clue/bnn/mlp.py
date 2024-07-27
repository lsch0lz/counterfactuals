from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, flatten_image):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.flatten_image = flatten_image

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.flatten_image:
            x = x.view(-1, self.input_dim)
        return self.block(x)
