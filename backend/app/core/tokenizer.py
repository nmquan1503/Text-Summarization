from .vocab import Vocab
import re
from typing import List, Tuple

class Tokenizer:
    def __init__(
        self, 
        vocab: Vocab,
        max_doc_length: int,
        max_sent_length: int,
    ):
        self.vocab = vocab
        self.max_doc_length = max_doc_length
        self.max_sent_length = max_sent_length

    def __call__(self, value: str | List):
        if isinstance(value, str):
            return self.tokenize(value)
        return [self.tokenize(doc) for doc in value]
    
    def decode(self, ids: List[int], oov: List[str]):
        ids = [id for id in ids if id not in [self.vocab.unk_id, self.vocab.pad_id, self.vocab.eos_id, self.vocab.sos_id, self.vocab.num_id, self.vocab.time_id, self.vocab.date_id]]
        words = []
        len_vocab = len(self.vocab)
        for id in ids:
            if id >= len_vocab + len(oov):
                continue
            elif id >= len_vocab:
                words.append(oov[id - len_vocab])
            else:
                words.append(self.vocab.get_word(id))
        return ' '.join(words)

    def tokenize(self, doc: str) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[str]]:
        origin, extend, oov = self.encode(doc)
        origin = self.padding(origin)
        extend = self.padding(extend)
        mask = [[1 if token != self.vocab.pad_id else 0 for token in sent] for sent in origin]
        return origin, extend, mask, oov

    def encode(self, doc: str):
        parts = re.split(r'\s([.!?:])(?:\s+|$)', doc.strip())
        parts = [part for part in parts if part]
        sentences = []
        buffer = ''
        for part in parts:
            if part.strip() in '.!?"\'':
                buffer += ' ' + part.strip()
            else:
                if buffer:
                    sentences.append(buffer)
                buffer = part.strip()
        if buffer:
            sentences.append(buffer.strip())
        tokens = [sentence.split() for sentence in sentences]
        if len(tokens) > self.max_doc_length:
            tokens = tokens[:self.max_doc_length]
        origin = []
        extend = []
        oov = []
        for sentence in tokens:
            origin_sent = []
            extend_sent = []
            if len(sentence) > self.max_sent_length:
                sentence = sentence[:self.max_sent_length]
            for word in sentence:
                id = self.vocab.get_index(word)
                if id == self.vocab.unk_id:
                    if re.fullmatch(self.vocab.num_regexp, word):
                        id = self.vocab.num_id
                    elif re.fullmatch(self.vocab.time_regexp, word):
                        id = self.vocab.time_id
                    elif re.fullmatch(self.vocab.date_regexp, word):
                        id = self.vocab.date_id
                    origin_sent.append(id)
                    if word not in oov:
                        oov.append(word)
                    extend_sent.append(len(self.vocab) + oov.index(word))
                else:
                    origin_sent.append(id)
                    extend_sent.append(id)
            origin.append(origin_sent)
            extend.append(extend_sent)
        return origin, extend, oov

    def padding(self, doc: str):
        padded = []
        if len(doc) > self.max_doc_length:
            doc = doc[:self.max_doc_length]
        for sentence in doc:
            if len(sentence) < self.max_sent_length:
                sentence += [self.vocab.pad_id] * (self.max_sent_length - len(sentence))
            elif len(sentence) > self.max_sent_length:
                sentence = sentence[:self.max_sent_length]
            padded.append(sentence)
        if len(padded) < self.max_doc_length:
            padded += [[self.vocab.pad_id] * self.max_sent_length] * (self.max_doc_length - len(padded))

        return padded