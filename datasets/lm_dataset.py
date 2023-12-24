from typing import Any
import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    """
    Dataset for language model.

    Args:
        data (list[dict[str, Any]]): The corpus on which the language model will be trained on.
        device (torch.device): The device where the calculations will be done, i.e. CPU or CUDA.
    """
    def __init__(self, data: list[dict[str, Any]], device: torch.device) -> None:
        """
        Initialize the dataset.

        Args:
            data (list[dict[str, Any]]): The corpus on which the language model will be trained on.
            device (torch.device): The device where the calculations will be done, i.e. CPU or CUDA.
        """
        self._corpus: list[list[int]] = [text['encoded'] for text in data]
        self._device = device

    def __len__(self)  -> int:
        """
        Returns the length of the dataset.
        """
        return len(self._corpus)

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """
        Returns the example at index idx. The token sequence will get start, end and padding tokens
        so that the sequence have the length context_length.

        Args:
            idx (int): The index of the example to be retrieved.
        """
        encoded_text: list[int] = self._corpus[idx]

        x = torch.tensor(encoded_text[:-1], dtype=torch.long).to(self._device)
        y = torch.tensor(encoded_text[1:], dtype=torch.long).to(self._device)

        return x, y