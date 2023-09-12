from .tokenizer import Tokenizer
from typing import List

class CharacterTokenizer(Tokenizer):
    """
    Tokenizer that tokenizes input text into individual characters.
    """

    def fit(self, corpus: List[str]) -> None:
        """
        This method is not used for character tokenization and does nothing.
        """
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text into a list of individual characters.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of individual characters from the input text.
        """
        return [char for char in list(text)]