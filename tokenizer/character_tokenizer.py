from tokenizer import Tokenizer
from typing import List

class CharacterTokenizer(Tokenizer):
    def fit(self, corpus: List[str]) -> None:
        pass

    def tokenize(self, text: str) -> List[str]:
        return [char for char in list(text)]