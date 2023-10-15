from typing import Callable, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from unidecode import unidecode

import torch

from tokenizer.tokenizer import Tokenizer

def create_pipeline(tokenizer: Tokenizer, train_size: float) -> Pipeline:
    """
    Creates a pipeline that takes text as input and produces the train and test data for the language model.

    Args:
        tokenizer (Tokenizer): A text tokenizer.
        train_size (float): Represents the propotion of the dataset to include in the train data set. Should be a float between 0 and 1.

    Returns:
        Pipeline: A pipeline transforming the text into tensors and split the data into two sets.
    """
    assert 0 <= train_size <= 1, "train_size must be between 0 and 1"
    return Pipeline([
        ('unicode_to_ascii', FunctionTransformer(unidecode)),
        ('encode', FunctionTransformer(tokenizer.encode)),
        ('create_tensor', FunctionTransformer(tokens_to_tensor)),
        ('train_test_split', FunctionTransformer(create_data_splitter(train_size)))
    ])

def tokens_to_tensor(tokens: List[int]) -> torch.tensor:
    """
    Transforms a list of tokens into a tensor.
    
    Args:
        tokens (List[int]): A list containing numeric representations of the tokens.

    Returns:
        torch.tensor: Returns a tensor containing the token encodings.
    """
    return torch.tensor(tokens, dtype=torch.long)

def create_data_splitter(train_size: float) -> Callable[[torch.tensor], Tuple[torch.tensor, torch.tensor]]:
    """
        Creates a function which splits the data with the given train_size.

        Args:
            train_size (float): Represents the propotion of the dataset to include in the train data set. Should be a float between 0 and 1.

        Returns:
            Callable[[torch.tensor], Tuple[torch.tensor, torch.tensor]]: Returns a function which splits the given data into two sets.
    """
    def split_data(data: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
            Function that splits the given text into two sets.

            Args:
                data (torch.tensor): The data which should be split into two sets.

            Returns:
                Tuple[torch.tensor, torch.tensor]: The splitted sets.
        """
        n = int(train_size * len(data))
        return data[:n], data[n:]

    return split_data