from .tokenizer import Tokenizer
from typing import Dict, List, Tuple
import re
from collections import defaultdict

class BytePairEncodingTokenizer(Tokenizer):
    """
    Tokenizer using Byte Pair Encoding (BPE) algorithm.

    Args:
        num_merges (int): The number of BPE merges to perform during training.

    Attributes:
        num_merges (int): The number of BPE merges to perform during training.
    """

    def __init__(self, num_merges: int) -> None:
        """
        Initialize the BytePairEncodingTokenizer.

        Args:
            num_merges (int): The number of BPE merges to perform during training.
        """
        super().__init__()
        self.num_merges = num_merges

    def get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """
        Build the initial vocabulary from a given corpus.

        Args:
            corpus (List[str]): A list of input text documents.

        Returns:
            Dict[str, int]: A dictionary mapping tokenized words to their frequencies.
        """
        vocab = defaultdict(int)

        for document in corpus:
            for word in document.split():
                char_split = ' '.join(word)
                vocab[char_split] += 1

        return dict(vocab)

    def get_pair_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Calculate statistics for pairs of tokens in the vocabulary.

        Args:
            vocab (Dict[str, int]): A dictionary mapping tokenized words to their frequencies.

        Returns:
            Dict[Tuple[str, str], int]: A dictionary mapping token pairs to their frequencies.
        """
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq

        return dict(pairs)

    def merge_vocab(self, best_pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
        """
        Merge the vocabulary by replacing the best pair with a single token.

        Args:
            best_pair (Tuple[str, str]): The token pair to merge.
            vocab_in (Dict[str, int]): The input vocabulary.

        Returns:
            Dict[str, int]: The updated vocabulary with merged tokens.
        """
        pattern = re.escape(' '.join(best_pair))
        replacement = ''.join(best_pair)

        vocab_out = {re.sub(pattern, replacement, word_in): freq for word_in, freq in vocab_in.items()}
        return vocab_out

    def fit(self, corpus: List[str]) -> None:
        """
        Train the Byte Pair Encoding tokenizer on a given corpus.

        Args:
            corpus (List[str]): A list of input text documents.
        """
        vocab = self.get_vocab(corpus)

        for i in range(self.num_merges):
            pair_stats = self.get_pair_stats(vocab)
            if not pair_stats:
                break

            best_pair = max(pair_stats, key=pair_stats.get)
            new_token = best_pair[0] + best_pair[1]
            self.stoi[new_token] = i + self.vocab_size
            self.itos[i + self.vocab_size] = new_token

            vocab = self.merge_vocab(best_pair, vocab)

        self.vocab_size = len(self.stoi)

    def create_new_word(self, word: List[str], pair_to_merge: Tuple[str, int]) -> List[str]:
        """
        Create a new word by merging a specific token pair.

        Args:
            word (List[str]): The list of tokens representing the current word.
            pair_to_mrge (Tuple[str,int]): The character pair to merge.

        Returns:
            List[str]: The word with the specified pair merged.
        """
        i = 0
        while i < len(word) - 1:
            token_1 = word[i]
            token_2 = word[i + 1]
            if (token_1 + token_2 == pair_to_merge[0]):
                word[i] = token_1 + token_2
                del word[i + 1]

            i += 1

        return word

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a given text using the trained Byte Pair Encoding tokenizer.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokenized subwords.
        """
        word = list(text)

        while True:
            pairs = [word[i] + word[i + 1] for i in range(len(word) - 1)]
            bpe_code_pairs = sorted([(pair, self.stoi[pair]) for pair in pairs if pair in self.stoi], key=lambda x: x[1])
            if not bpe_code_pairs:
                break

            pair_to_merge = bpe_code_pairs[0]

            word = self.create_new_word(word, pair_to_merge)

        return word
