from abc import ABC, abstractmethod
import pickle

from tokenizer import special_tokens

BASE_VOCAB = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~\n'

class Tokenizer(ABC):
    """
    Abstract base class for tokenization and encoding of text data.

    Attributes:
        stoi (dict[str, int]): A dictionary mapping tokens to their corresponding integer indices.
        itos (dict[int, str]): A dictionary mapping integer indices to their corresponding tokens.
    """

    def __init__(self):
        """
        Initialize a Tokenizer instance with a base vocabulary.
        """
        self.stoi: dict[str, int] = {
            special_tokens.SpecialTokens.PAD: 0,
            special_tokens.SpecialTokens.SOS: 1,
            special_tokens.SpecialTokens.EOS: 2
        }

        current_vocab_size = len(self.stoi)

        for i, char in enumerate(list(BASE_VOCAB)):
            self.stoi[char] = i + current_vocab_size

        self.itos: dict[int, str] = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        """
        Returns size of vocabulary.
        """
        return len(self.stoi)

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """
        Abstract method to fit the tokenizer on a corpus of text data.

        Args:
            corpus (list[str]): A list of text string to train the tokenizer on.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Abstract method to tokenize a given text into a list of tokens.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str]: A list of tokens extracted from the input text.
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encodes a given text into a list of integer indices based on the tokenizer's vocabulary.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of integer indices corresponding to the tokens in the input text.
        """
        tokenization: list[str] = self.tokenize(text)
        return [self.stoi[token] for token in tokenization]

    def decode(self, encoding: list[int]) -> str:
        """
        Decodes a list of integer indices into the original text using the tokenizer's vocabulary.

        Args:
            encoding (list[int]): A list of integer indices to decode.

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
