from tokenizer import Tokenizer
from typing import Dict, List, Tuple
import re
from collections import defaultdict

class BytePairEncoderTokenizer(Tokenizer):
    def __init__(self, num_merges: int) -> None:
        super().__init__()
        self.num_merges = num_merges

    def get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        vocab = defaultdict(int)

        for document in corpus:
            for word in document.split():
                char_split = ' '.join(word)
                vocab[char_split] += 1

        return dict(vocab)

    def get_pair_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq

        return dict(pairs)

    def merge_vocab(self, best_pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
        pattern = re.escape(' '.join(best_pair))
        replacement = ''.join(best_pair)

        vocab_out = {re.sub(pattern, replacement, word_in): freq for word_in, freq in vocab_in.items()}
        return vocab_out

    def fit(self, corpus: List[str]) -> None:
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
        word = list(text)

        while True:
            pairs = [word[i] + word[i + 1] for i in range(len(word) - 1)]
            bpe_code_pairs = sorted([(pair, self.stoi[pair]) for pair in pairs if pair in self.stoi], key=lambda x: x[1])
            if not bpe_code_pairs:
                break

            pair_to_merge = bpe_code_pairs[0]

            word = self.create_new_word(word, pair_to_merge)

        return word
