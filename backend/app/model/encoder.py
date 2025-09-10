import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .attention_pooling import AttentionPooling
from .residual_block import ResidualBlock


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size_word, 
        word_residual_configs,
        hidden_size_sent,
        sent_residual_configs,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size_word = hidden_size_word
        self.hidden_size_sent = hidden_size_sent
        self.dropout = nn.Dropout(dropout)

        self.word_input_norm = nn.LayerNorm(embedding_dim)
        
        # Word-level BiLSTM
        self.word_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size_word,
            bidirectional=True,
            batch_first=True
        )

        self.word_attention_pooling = AttentionPooling(
            2 * self.hidden_size_word, 
        )

        self.word_residuals = nn.Sequential(
            *[
                ResidualBlock(
                    input_dim=2 * hidden_size_word,
                    hidden_dim=config['hidden_dim'],
                    activation=config['activation']
                )
                for config in word_residual_configs
            ]
        )

        self.sent_input_norm = nn.LayerNorm(2 * hidden_size_word)
        
        # Sentence-level BiLSTM
        self.sent_layer = nn.LSTM(
            input_size=self.hidden_size_word * 2,
            hidden_size=self.hidden_size_sent,
            bidirectional=True,
            batch_first=True
        )

        self.sent_attention_pooling = AttentionPooling(
            2 * self.hidden_size_sent, 
            # dropout
        )
        
        self.sent_residuals = nn.Sequential(
            *[
                ResidualBlock(
                    input_dim=2 * self.hidden_size_sent,
                    hidden_dim=config['hidden_dim'],
                    activation=config['activation']
                )
                for config in sent_residual_configs
            ]
        )
        
    def forward(self, embedded_inputs, attention_masks, debug=False):
        """
        B: Batch size
        S: Number of sentences
        W: Number of words in a sentences
        D: Embedding dim
        HW: Hidden word size
        HS: Hidden sent size

        Args:
            embedded_inputs: [B, S, W, D]
            attention_mask: [B, S, W]
        Returns:
            output: [B, 2HS]
            word_layer_outputs: [B, S * W, 2HW]
        """

        B, S, W = attention_masks.shape
        device = embedded_inputs.device

        # Flatten 
        # [B, S, W, D] -> [B * S, W, D]
        flatted_inputs = embedded_inputs.view(B * S, W, -1)
        flatted_inputs = self.word_input_norm(flatted_inputs)
        flatted_inputs = self.dropout(flatted_inputs)

        # [B, S, W] -> [B * S, W]
        flatted_masks = attention_masks.view(B * S, -1)

        # === Word-level BiLSTM ===
        
        # Compute lengths for packing
        sent_lengths = flatted_masks.sum(dim=1).cpu()
        valid_masks = sent_lengths > 0

        packed_word_layer_inputs = pack_padded_sequence(
            flatted_inputs[valid_masks],
            lengths=sent_lengths[valid_masks],
            batch_first=True,
            enforce_sorted=False
        )
        packed_word_layer_outputs, _ = self.word_layer(packed_word_layer_inputs)
        
        # [B_valid, W, 2HW]
        unpacked_word_layer_outputs, _ = pad_packed_sequence(packed_word_layer_outputs, batch_first=True, total_length=W)

        # [B * S, W, 2HW]
        word_layer_outputs = torch.zeros(B * S, W, 2 * self.hidden_size_word, device=device)
        word_layer_outputs[valid_masks] = unpacked_word_layer_outputs

        # [B * S, W, 2HW] -> [B * S, 2HW]
        sent_layer_inputs = self.word_attention_pooling(word_layer_outputs, flatted_masks)
        sent_layer_inputs = self.dropout(sent_layer_inputs)
        
        sent_layer_inputs = self.word_residuals(sent_layer_inputs)

        # [B * S, 2HW] -> [B, S, 2HW]
        sent_layer_inputs = sent_layer_inputs.view(B, S, -1)
        sent_layer_inputs = self.sent_input_norm(sent_layer_inputs)
        sent_layer_inputs = self.dropout(sent_layer_inputs)

        # === Sentence-level BiLSTM ===

        # Compute lengths for packing
        doc_masks = (attention_masks.sum(dim=2) > 0).long()
        doc_lengths = doc_masks.sum(dim=-1).cpu()

        packed_sent_layer_inputs = pack_padded_sequence(
            sent_layer_inputs,
            lengths=doc_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_sent_layer_outputs, _ = self.sent_layer(packed_sent_layer_inputs)

        sent_layer_outputs, _ = pad_packed_sequence(packed_sent_layer_outputs, batch_first=True, total_length=S)

        # [B, S, 2HS] -> [B, 2HS]
        outputs = self.sent_attention_pooling(sent_layer_outputs)
        outputs = self.dropout(outputs)
        
        outputs = self.sent_residuals(outputs)
        
        return outputs, word_layer_outputs.view(B, S * W, -1)