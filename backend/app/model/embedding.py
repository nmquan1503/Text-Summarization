from ..core.vocab import Vocab
import torch
import torch.nn as nn
import numpy as np
import torch.functional as F

class Word2VecEmbedding(nn.Module):
    def __init__(self, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), vocab.embedding_dim)
        self.adapter = nn.Linear(vocab.embedding_dim, vocab.embedding_dim)
        self.load_pretrained_weights()
    
    def load_pretrained_weights(self):
        self.embedding.weight.requires_grad = False
        weights = []
        for i in range(len(self.vocab)):
            word = self.vocab.get_word(i)
            if word in self.vocab.model.wv:
                vec = self.vocab.model.wv[word]
            else:
                vec = np.random.normal(scale=0.1, size=self.vocab.embedding_dim)
            weights.append(torch.tensor(vec, dtype=torch.float32))
        weights[self.vocab.pad_id] = torch.zeros(self.vocab.embedding_dim)
        
        weights_tensor = torch.stack(weights)
        with torch.no_grad():
            self.embedding.weight.copy_(weights_tensor)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.adapter(x)
        x = F.gelu(x)
        return x