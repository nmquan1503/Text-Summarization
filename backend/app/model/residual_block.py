import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation="ReLU",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()

    def forward(self, x):
        x = self.norm(x)
        out = self.fc1(x)
        out = self.activation(out)

        out = self.fc2(out)
        return self.activation(x + out)