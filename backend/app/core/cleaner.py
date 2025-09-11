import unicodedata
import re
from underthesea import word_tokenize
from typing import List

class Cleaner:
    def __init__(self):
        pass

    def __call__(self, value: str | List[str]):
        if isinstance(value, str):
            return self.clean_text(value)
        return [self.clean_text(text) for text in value]
    
    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = text.lower()
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        text = text.replace('`', "'")
        text = text.replace('…', '.')
        text = re.sub(r'([.,?!\'"])\1{1,}', r'\1', text)
        text = re.sub(r'[‐‑‒–—−]', '-', text)
        text = re.sub(r'(\d{1,2})h(\d{1,2})p\b', r'\1h\2', text)
        text = re.sub(r'(\d{1,2})h(\d{1,2})\'\b', r'\1h\2', text)
        text = re.sub(r'(\d{1,2}):(\d{1,2})\b', r'\1h\2', text)
        text = text.replace('-', ' - ')
        UNITS = [
            "m", "km", "cm", "mm", "μm",
            "g", "kg", "mg", "lb",
            "s", "ms", "μs", "ns",
            "hz", "khz", "mhz", "ghz",
            "b", "kb", "mb", "gb", "tb",
            "v", "w", "kw", "mah",
            "°c", "°f", "l", "ml",
            "nm", "lux", "rpm", "km/h", "m/s", "nits", "%"
        ]
        for unit in UNITS:
            pattern = rf"(\d+(?:[\.,]\d+)?){unit}"
            repl = rf"\1 {unit}"
            text = re.sub(pattern, repl, text)
        text = re.sub(r"\s+", ' ', text)
        text = word_tokenize(text, format='text')
        text = text.replace('km / h', 'km/h')
        text = text.replace('m / s', 'm/s')
        return text.strip()