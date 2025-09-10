import torch
import torch.nn as nn
import torch.functional as F
from .embedding import Word2VecEmbedding
from .encoder import Encoder
from .decoder import Decoder
from ..core.vocab import Vocab

class Model(nn.Module):
    def __init__(
            self,
            vocab: Vocab,
            enc_hidden_size_word,
            enc_word_residual_configs,
            enc_hidden_size_sent,
            enc_sent_residual_configs,
            dec_hidden_size,
            dec_residual_configs,
            dec_attn_dim,
            device: torch.device,
            dropout=0.3
    ):
        super().__init__()
        self.vocab = vocab

        # === Embedding ===
        self.embedding_layer = Word2VecEmbedding(vocab)

        # === Encoder ===
        self.encoder = Encoder(
            embedding_dim=vocab.embedding_dim,
            hidden_size_word=enc_hidden_size_word,
            word_residual_configs=enc_word_residual_configs,
            hidden_size_sent=enc_hidden_size_sent,
            sent_residual_configs=enc_sent_residual_configs,
            dropout=dropout
        )

        # === Adapter ===
        self.dropout = nn.Dropout(dropout)
        self.adapter = nn.Linear(2 * enc_hidden_size_sent, dec_hidden_size)

        # === Decoder ===
        self.decoder = Decoder(
            embedding_dim=vocab.embedding_dim,
            embedding_matrix=self.embedding_layer.embedding.weight,
            hidden_size=dec_hidden_size,
            vocab_size=len(vocab),
            hidden_size_word=enc_hidden_size_word,
            residual_configs=dec_residual_configs,
            attn_dim=dec_attn_dim,
            dropout=dropout
        )
        
        self.to(device)
    
    def forward(
        self, 
        input_ids, 
        ext_input_ids, 
        target_ids, 
        attention_mask, 
        teacher_forcing_ratio=0.5, 
        debug=False
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
        L: Target length
        
        Args:
            input_ids: [B, S, W]
            ext_input_ids: [B, S, W]
            target_ids: [B, L]
            attention_mask: [B, S, W]
        Returns:
            final_dists: [B, L, EV]
        """
        B, T = target_ids.size()
        pad_id = self.vocab.pad_id
        
        target_mask = target_ids != pad_id
        max_len = target_mask.sum(dim=1).max().item()

        # === Encode ===

        # [B, S, W] -> [B, S, W, D]
        enc_embedded_input = self.embedding_layer(input_ids)
        
        # [B, 2HS], [B, S * W, 2HW]
        sent_hidden, word_outputs = self.encoder(enc_embedded_input, attention_mask, debug=debug)

        # Precompute encoder attention projection

        # [B, S * W, 2HW] -> [B, S * W, A]
        enc_proj = self.decoder.attention.enc_proj(word_outputs)
        enc_proj = self.decoder.attention.enc_norm(enc_proj)
        
        # === Init decoder state ===
        
        # [B]
        decoder_input = torch.full((B,), self.vocab.sos_id, dtype=torch.long, device=input_ids.device)
        
        # [B] -> [B, D]
        decoder_embedded_input = self.embedding_layer(decoder_input)
        
        # [B, 2HS] -> [B, H] -> [1, B, H]
        h = self.adapter(sent_hidden).unsqueeze(0)
        h = F.relu(h)
        h = self.dropout(h)
        c = torch.zeros_like(h)
        
        decoder_state = (h, c)
        
        # [B, S, W] -> [B, S * W]
        input_ids = input_ids.view(B, -1)
        ext_input_ids = ext_input_ids.view(B, -1)

        # [B, S, W] -> [B, S * W]
        encoder_mask = attention_mask.view(B, -1)

        ext_vocab_size = ext_input_ids.max().item() + 1

        # === Decode ===
        final_dists = torch.zeros(B, T, ext_vocab_size, device=input_ids.device, dtype=torch.float32)
        for t in range(max_len):

            # [B, EV], ([1, B, H], [1, B, H])
            final_dist, decoder_state = self.decoder(
                decoder_embedded_input, 
                decoder_state, 
                word_outputs,
                ext_input_ids, 
                ext_vocab_size,
                encoder_mask=encoder_mask,
                debug=debug and (t % 20 == 0),
                enc_proj=enc_proj
            )

            final_dists[:, t, :] = final_dist

            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_ids[:, t]
                decoder_embedded_input = self.embedding_layer(decoder_input)
            else:
                decoder_input = final_dist.argmax(1)
                oov_mask = decoder_input >= len(self.vocab)
                if oov_mask.any():
                    oov_tokens = decoder_input[oov_mask]
                    token_pos = (ext_input_ids[oov_mask] == oov_tokens.unsqueeze(1)).float().argmax(dim=1)
                    copied_token = input_ids[oov_mask, token_pos]
                    decoder_input[oov_mask] = copied_token
                decoder_embedded_input = self.embedding_layer(decoder_input)

        return final_dists