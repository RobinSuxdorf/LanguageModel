import tokenizer

class CharacterTokenizer(tokenizer.Tokenizer):
    """
    Tokenizer that tokenizes input text into individual characters.
    """

    def fit(self, corpus: list[str]) -> None:
        """
        This method is not used for character tokenization and does nothing.
        """
        pass

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text into a list of individual characters.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str]: A list of individual characters from the input text.
        """
        return [char for char in list(text)]