import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .pointer_generator import PointerGenerator
from .residual_block import ResidualBlock

class Decoder(nn.Module):
    def __init__(
        self, 
        embedding_dim, 
        embedding_matrix,   
        hidden_size, 
        vocab_size,
        hidden_size_word, 
        residual_configs,
        attn_dim=256, 
        dropout=0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size_word=hidden_size_word
        self.embedding_matrix = embedding_matrix

        self.dropout = nn.Dropout(dropout)
        
        # === Attention ===

        # Project encoder word_outputs
        self.attention = Attention(
            hidden_size_word=hidden_size_word,
            hidden_size_decoder=hidden_size,
            attention_dim=attn_dim,
            dropout=dropout
        )

        self.lstm_input_norm = nn.LayerNorm(embedding_dim + 2 * hidden_size_word)

        # === Decoder LSTM ===
        self.dec_lstm = nn.LSTM(
            input_size=self.embedding_dim + 2 * hidden_size_word,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.residuals = nn.Sequential(
            *[
                ResidualBlock(
                    input_dim=hidden_size,
                    hidden_dim=config['hidden_dim'],
                    # dropout=dropout,
                    activation=config['activation']
                )
                for config in residual_configs
            ]
        )

        # === Vocab projection ===
        self.contextual_dec_out_norm = nn.LayerNorm(hidden_size + 2 * hidden_size_word)
        self.emb_proj = nn.Linear(hidden_size + 2 * hidden_size_word, self.embedding_dim)
        
        # === Pointer-generator gate ===
        self.ptr_gen = PointerGenerator(
            hidden_size_decoder=self.hidden_size,
            hidden_size_word=self.hidden_size_word,
            embedding_dim=self.embedding_dim
        )
        
    def forward(
        self, 
        embedded_input, 
        decoder_state, 
        encoder_word_outputs, 
        ext_input_ids, 
        ext_vocab_size, 
        encoder_mask=None,
        debug=False,
        enc_proj=None
    ):
        """
        B: Batch size
        S: Number of sentences
        W: Number of words in a sentences
        D: Embedding dim
        HW: Hidden word size
        HS: Hidden sent size
        H: Hidden decoder size
        A: Attention dim
        V: Vocab size
        EV: Extended vocab size
        
        Args:
            embedded_input: [B, D]
            decoder_state: ([1, B, H], [1, B, H])
            encoder_word_outputs: [B, S * W, 2HW]
            ext_input_ids: [B, S * W]
            ext_vocab_size: int
        Returns:
            final_dist: [B, EV]
            next_decoder_state: ([1, B, H], [1, B, H])
        """
        B, SxW, _ = encoder_word_outputs.size()

        # === Embedding input token ===
        
        # [B, D] -> [B, 1, D]
        embedded_input = embedded_input.unsqueeze(1)

        # === Attention ===
        # [1, B, H] -> [B, 1, H]
        dec_hidden = decoder_state[0].transpose(0, 1)

        # [B, 2HW], [B, S * W]
        context, attn_weights = self.attention(dec_hidden, encoder_word_outputs, enc_proj, encoder_mask)

        # [B, 2HW] -> [B, 1, 2HW]
        context = context.unsqueeze(1)

        # === Decoder LSTM ===
        
        # [B, 1, D] cat [B, 1, 2HW] -> [B, 1, D + 2HW]
        lstm_input = torch.cat([embedded_input, context], dim=-1)
        lstm_input = self.lstm_input_norm(lstm_input)
        lstm_input = self.dropout(lstm_input)

        lstm_out, next_decoder_state = self.dec_lstm(lstm_input, decoder_state)
        lstm_out = self.residuals(lstm_out)

        # === Vocab distribution ===
        
        # [B, 1, H] cat [B, 1, 2HW] -> [B, 1, H + 2HW]
        contextual_dec_out = torch.cat([lstm_out, context], dim=-1)
        contextual_dec_out = self.contextual_dec_out_norm(contextual_dec_out)
        contextual_dec_out = self.dropout(contextual_dec_out)

        # [B, 1, H + 2HW] -> [B, 1, D] -> [B, D]
        emb_out = self.emb_proj(contextual_dec_out).squeeze(1)
        
        # [B, D] x [D, V] -> [B, V]
        vocab_logits = torch.matmul(emb_out, self.embedding_matrix.T)
        vocab_dist = F.softmax(vocab_logits, dim=-1)

        # === Pointer generator ===

        # [B, 1, H + 2HW] -> [B, H + 2HW]
        contextual_dec_out = contextual_dec_out.squeeze(1)

        # [B, 1, D] -> [B, D]
        embedded_input = embedded_input.squeeze(1)
        
        final_dist = self.ptr_gen(contextual_dec_out, embedded_input, vocab_dist, attn_weights, ext_input_ids, ext_vocab_size)

        return final_dist, next_decoder_state