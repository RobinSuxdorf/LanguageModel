from abc import ABC, abstractmethod
from typing import Dict, List
import pickle

BASE_VOCAB = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\n'

class Tokenizer(ABC):
    """
    Abstract base class for tokenization and encoding of text data.

    Attributes:
        stoi (Dict[str, int]): A dictionary mapping tokens to their corresponding integer indices.
        itos (Dict[int, str]): A dictionary mapping integer indices to their corresponding tokens.
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.
    """

    def __init__(self):
        """
        Initialize a Tokenizer instance with a basse vocabulary.
        """
        self.stoi: Dict[str, int] = {char: i for i, char in enumerate(list(BASE_VOCAB))}
        self.itos: Dict[int, str] = {i: char for i, char in enumerate(list(BASE_VOCAB))}
        self.vocab_size = len(self.stoi)

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """
        Abstract method to fit the tokenizer on a corpus of text data.

        Args:
            corpus (List[str]): A list of text string to train the tokenizer on.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Abstract method to tokenize a given text into a list of tokens.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens extracted from the input text.
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a list of integer indices based on the tokenizer's vocabulary.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of integer indices corresponding to the tokens in the input text.
        """
        tokenization: List[str] = self.tokenize(text)
        return [self.stoi[token] for token in tokenization]

    def decode(self, encoding: List[int]) -> str:
        """
        Decodes a list of integer indices into the original text using the tokenizer's vocabulary.

        Args:
            encoding (List[int]): A list of integer indices to decode.

        Returns:
            str: The decoded text.
        """
        return ''.join(self.itos[i] for i in encoding)

    def to_pkl(self, path: str) -> None:
        """
        Saves the tokenizer's state to a binary file.

        Args:
            path (str): The path of the file to save the tokenizer's state to.
        """
        with open(path, "wb") as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def read_pkl(cls, path: str):
        """
        Reads the tokenizer's state from a binary file.

        Args:
            path (str): The path of the file to load the tokenizer's state from.
        """
        try:
            with open(path, "rb") as file:
                tokenizer = cls.__new__(cls)
                tokenizer.__dict__ = pickle.load(file)
                return tokenizer
        except FileNotFoundError:
            print(f"File '{path}' not found.")
        except EOFError:
            print(f"Error while reading '{path}'. File may be corrupted.")
