from .cleaner import Cleaner
from .tokenizer import Tokenizer
from typing import Dict, Any, List, Union
from .vocab import Vocab
from ..model.model import Model
from .post_processor import PostProcessor
import json
import torch

class ModelWrapper:
    def __init__(
        self,
        vocab_path: str,
        model_config_path: str,
        model_weights_path: str,
        max_doc_length: int,
        max_sent_length: int,
        max_output_length: int,
        beam_size: int
    ):
        self.vocab = Vocab(vocab_path)
        self.cleaner = Cleaner()
        self.tokenizer = Tokenizer(
            vocab=self.vocab,
            max_doc_length=max_doc_length,
            max_sent_length=max_sent_length
        )
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            device = model_config['device']
            device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.model = Model(
            vocab=self.vocab,
            enc_hidden_size_word=model_config['enc_hidden_size_word'],
            enc_hidden_size_sent=model_config['enc_hidden_size_sent'],
            enc_word_residual_configs=model_config['enc_word_residual_configs'],
            enc_sent_residual_configs=model_config['enc_sent_residual_configs'],
            dec_hidden_size=model_config['dec_hidden_size'],
            dec_residual_configs=model_config['dec_residual_configs'],
            dec_attn_dim=model_config['dec_attn_dim'],
            device=device
        )
        model_state_dict = torch.load(model_weights_path, map_location=device)['model_state_dict']
        self.model.load_state_dict(model_state_dict)
        self.max_output_length = max_output_length
        self.beam_size = beam_size
        self.post_processor = PostProcessor
    
    def predict(self, value: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(value, str):
            inputs = [value]
        else:
            inputs = value
        
        cleaned = self.cleaner(inputs)
        tokenized = self.tokenizer(cleaned)
        input_ids = torch.tensor([item[0] for item in tokenized])
        ext_input_ids = torch.tensor([item[1] for item in tokenized])
        masks = torch.tensor([item[2] for item in tokenized])
        oovs = [item[3] for item in tokenized]
    
        outputs = self.model.generate(input_ids, ext_input_ids, masks, max_len=self.max_output_length, beam_size=self.beam_size)

        seqs = []
        for pred, oov in zip(outputs, oovs):
            seq = self.tokenizer.decode(pred, oov)
            seqs.append(seq)
        
        seqs = self.post_processor(seqs)

        if isinstance(value, str):
            return seqs[0]
        return seqs
