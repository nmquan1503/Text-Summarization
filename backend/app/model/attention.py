import torch
import torch.nn as nn
import torch.functional as F

class Attention(nn.Module):
    def __init__(
        self,
        hidden_size_word,
        hidden_size_decoder,
        attention_dim,
        dropout
    ):
        super().__init__()
        self.enc_proj = nn.Linear(hidden_size_word * 2, attention_dim)
        self.enc_norm = nn.LayerNorm(attention_dim)
        self.dec_proj = nn.Linear(hidden_size_decoder, attention_dim, bias=False)
        self.dec_norm = nn.LayerNorm(attention_dim)
        self.score_proj = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.score_proj.bias, 0.1)
    
    def forward(self, dec_hidden, enc_outputs, enc_proj=None, enc_mask=None):
        """
        Args:
            dec_hidden: [B, 1, H]
            enc_outputs: [B, S * W, 2HW]
            enc_proj: [B, S * W, A]
            enc_mask: [B, S * W]
        Returns:
            context: [B, 2HW]
        """
        B, SxW, _ = enc_outputs.size()

        if enc_proj is None:
            # [B, S * W, 2HW] -> [B, S * W, A]
            enc_proj = self.enc_proj(enc_outputs)
            enc_proj = self.enc_norm(enc_proj)

        # [B, 1, H] -> [B, 1, A]
        dec_proj = self.dec_proj(dec_hidden)
        dec_proj = self.dec_norm(dec_proj)

        attn_features = F.gelu(enc_proj + dec_proj)
        attn_features = self.dropout(attn_features)

        # [B, S * W, A] -> [B, S * W, 1] -> [B, S * W]
        attn_scores = self.score_proj(attn_features).squeeze(-1)

        if enc_mask is not None:
            attn_scores = attn_scores.masked_fill(enc_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # [B, 1, S * W] @ [B, S * W, 2HW] -> [B, 1, 2HW] -> [B, 2HW]
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)

        return context, attn_weights