import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = activation()

    def forward(self, x):
        x = self.norm(x)
        out = self.fc1(x)
        out = self.activation(out)

        out = self.fc2(out)
        return self.activation(x + out)