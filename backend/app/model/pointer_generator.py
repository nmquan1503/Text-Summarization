import torch
import torch.nn as nn

class PointerGenerator(nn.Module):
    def __init__(
        self, 
        hidden_size_decoder, 
        hidden_size_word, 
        embedding_dim
    ):
        super().__init__()
        self.ptr_proj = nn.Linear(hidden_size_decoder + 2 * hidden_size_word + embedding_dim, 1)

    def forward(self, context, emb_input, vocab_dist, attn_weights, ext_input_ids, ext_vocab_size):
        """
        Args:
            context: [B, 2HW + H]
            embedded_input: [B, D]
            vocab_dist: [B, V]
            attn_weights: [B, S * W]
            ext_input_ids: [B]
            ext_vocab_size: int
        Returns:
            final_dist: [B, EV]
        """
        B, V = vocab_dist.size()
        device = context.device

        # [B, 2HW + H] cat [B, D] -> [B, 2HW + H + D]
        ptr_input = torch.cat([context, emb_input], dim=-1)

        # [B, 2HW + H + D] -> [B, 1]
        ptr_gate = torch.sigmoid(self.ptr_proj(ptr_input))

        # [B, 1] * [B, V] -> [B, V]
        vocab_dist_scaled = ptr_gate * vocab_dist

        # [B, 1] * [B, S * W] -> [B, S * W]
        attn_dist_scaled = (1 - ptr_gate) * attn_weights

        final_dist = torch.zeros(B, ext_vocab_size, device=device)
        final_dist[:, :vocab_dist.size(-1)] += vocab_dist_scaled
        final_dist.scatter_add_(1, ext_input_ids.long(), attn_dist_scaled)

        return final_dist