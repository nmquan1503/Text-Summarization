from gensim.models import Word2Vec
from typing import Tuple

class Vocab:
    def __init__(self, word2vec_model_path: str):
        self.model = Word2Vec.load(word2vec_model_path)
        
        self.word2id = {}
        self.id2word = {}
        self.embedding_dim = self.model.vector_size
        self.build()
        self.unk_id = self.word2id['<UNK>']
        self.pad_id = self.word2id['<PAD>']
        self.sos_id = self.word2id['<SOS>']
        self.eos_id = self.word2id['<EOS>']
        self.num_id = self.get_index('<NUM>')
        self.time_id = self.get_index('<TIME>')
        self.date_id = self.get_index('<DATE>')
        self.num_regexp = r"[\d.,]*\d[\d.,]*"
        self.time_regexp = r"(\d{1,2}h(\d{1,2})?)"
        self.date_regexp = r"(\d{1,2}/\d{1,2}(/\d{2,4})?)"

    def build(self):
        id = 0
        for word in self.model.wv.index_to_key:
            self.word2id[word] = id
            self.id2word[id] = word
            id += 1
        special_tokens = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
        for token in special_tokens:
            self.word2id[token] = id
            self.id2word[id] = token
            id += 1
    
    def __len__(self):
        return len(self.word2id)
    
    def get_index(self, word):
        return self.word2id.get(word, self.unk_id)
    
    def get_word(self, id):
        return self.id2word.get(id, '<UNK>')

    def decode(self, ids, oov):
        ids = [id for id in ids if id not in [self.unk_id, self.pad_id, self.eos_id, self.sos_id, self.num_id, self.time_id, self.date_id]]
        words = []
        len_vocab = self.__len__()
        for id in ids:
            if id >= len_vocab + len(oov):
                continue
            elif id >= len_vocab:
                words.append(oov[id - len_vocab])
            else:
                words.append(self.get_word(id))
        return ' '.join(words)