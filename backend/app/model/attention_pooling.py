import torch
import torch.nn as nn
import torch.functional as F

class AttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()
        self.adapter = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )
        nn.init.constant_(self.score_mlp[-1].bias, 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, hiddens, mask=None):
        h = self.adapter(hiddens)
        h = F.relu(h)
        h = self.norm(h)
        
        # [..., H] -> [..., 1] -> [...]
        scores = self.score_mlp(h).squeeze(-1) / self.temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # [...] -> [..., 1]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)

        context = torch.sum(h * weights, dim=-2)

        return context + 0.1 * hiddens.mean(dim=-2)