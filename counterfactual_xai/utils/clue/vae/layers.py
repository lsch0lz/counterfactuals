from torch import nn


class SkipConnection(nn.Module):
    """
    Skip-connection over the sequence of layers in the constructor.
    The module passes input data sequentially through these layers
    and then adds original data to the result.
    """

    def __init__(self, *args):
        super(SkipConnection, self).__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)


def preact_leaky_mlp_block(width):
    return SkipConnection(
        nn.LeakyReLU(),
        nn.BatchNorm1d(num_features=width),
        nn.Linear(width, width),
    )
