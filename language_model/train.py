from typing import Dict, Tuple

from dataclasses import dataclass

import torch

from .generation import LanguageModel

@dataclass
class TrainArgs:
    max_iters: int = 250
    eval_iters: int = 5
    eval_interval: int = 10
    learning_rate: int = 3e-4
    batch_size: int = 64

class ModelTrainer:
    """
    Class for training language model on given datasets.

    Args:
        args (TrainArgs): The training hyperparameters.
        model (LanguageModel): The language model which should be trained.
        train_data (torch.tensor): The training dataset.
        test_data (torch.tensor): The test dataset.
    """
    def __init__(self, args: TrainArgs, model: LanguageModel, train_data: torch.tensor, test_data: torch.tensor):
        self._max_iters = args.max_iters
        self._eval_iters = args.eval_iters
        self._eval_interval = args.eval_interval
        self._batch_size = args.batch_size

        self._train_data = train_data
        self._test_data = test_data

        self._model = model
        self._optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.learning_rate)

    def _get_batch(self, data: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Returns a batch from a dataset.

        Args:
            data (torch.tensor): The dataset from which the batch should be created, i.e. train or test.
        Returns:
            Tuple[torch.tensor, torch.tensor]: Returns a tuple consisting of input and target values.
        """
        assert len(data) - self._model.context_length >= 0, 'Length of data is shorter than context_length'

        idx = torch.randint(len(data) - self._model.context_length, (self._batch_size, ))
        x = torch.stack([data[i:i + self._model.context_length] for i in idx])
        y = torch.stack([data[i + 1:i + self._model.context_length + 1] for i in idx])
        x, y = x.to(self._model._device), y.to(self._model._device)
        return x, y

    @torch.no_grad()
    def _estimate_loss(self) -> Dict[int, torch.tensor]:
        """
        Estimates the loss of the model.

        Returns:
            Dict[int, torch.tensor]: Returns a dictionary containing the train and test loss of the model.
        """
        out = {}
        self._model.encoder.eval()
        for index, dataset in enumerate([self._train_data, self._test_data]):
            losses = torch.zeros(self._eval_iters)
            for k in range(self._eval_iters):
                x, y = self._get_batch(dataset)
                logits, loss = self._model.encoder(x, y)
                losses[k] = loss.item()
            out[index] = losses.mean()
        self._model.encoder.train()
        return out

    def train(self) -> None:
        """
        Trains the model on the given trainings dataset.
        """
        for iter in range(self._max_iters):
            if iter % self._eval_interval == 0 or iter == self._max_iters -1 :
                losses = self._estimate_loss()
                print(f"step {iter}: train loss {losses[0]:.4f}, test loss {losses[1]:.4f}")

            x_batch, y_batch = self._get_batch(self._train_data)

            logits, loss = self._model.encoder(x_batch, y_batch)
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()