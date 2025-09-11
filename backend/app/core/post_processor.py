import re
from typing import List, Union

class PostProcessor:
    def __init__(self):
        pass

    def __call__(self, value: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(value, str):
            return self.process(value)
        return [self.process(text) for text in value]

    def process(self, text: str) -> str:
        text = text.replace('_', ' ')
        text = re.sub(r'\s+([!.,@])', r'\1', text)
        text = re.sub(r" (['\"]) ", r'\1', text)
        text = re.sub(r'(^\s*[a-z])', lambda m: m.group(1).upper(), text)
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        return text