from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from datasets import lm_dataset
from language_model import generation

@dataclass
class TrainArgs:
    max_iters: int = 250
    eval_iters: int = 5
    eval_interval: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 64

class ModelTrainer:
    """
    Class for training language model on given datasets.

    Args:
        model (LanguageModel): The language model which should be trained.
        train_text (str): The training text.
        test_text (str): The test text.
        args (TrainArgs): The training hyperparameters.
    """
    def __init__(
        self, 
        model: generation.LanguageModel, 
        train_text: str, 
        test_text: str,
        args: TrainArgs = TrainArgs()
    ):
        self._max_iters = args.max_iters
        self._eval_iters = args.eval_iters
        self._eval_interval = args.eval_interval
        self._batch_size = args.batch_size

        train_data = lm_dataset.LMDataset(model.tokenizer, train_text, model.context_length)
        train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        self._train_data_iter = iter(train_data_loader)

        test_data = lm_dataset.LMDataset(model.tokenizer, test_text, model.context_length)
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        self._test_data_iter = iter(test_data_loader)

        self._model = model
        self._optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def _estimate_loss(self) -> dict[int, torch.tensor]:
        """
        Estimates the loss of the model.

        Returns:
            dict[int, torch.tensor]: Returns a dictionary containing the train and test loss of the model.
        """
        out = {}
        self._model.encoder.eval()
        for index, iter in enumerate([self._train_data_iter, self._test_data_iter]):
            losses = torch.zeros(self._eval_iters)
            for k, (x, y) in enumerate(iter):

                logits, loss = self._model.encoder(x, y)
                losses[k] = loss.item()
            out[index] = losses.mean()
        self._model.encoder.train()
        return out

    def train(self) -> None:
        """
        Trains the model on the given trainings dataset.
        """
        for iter, (x_batch, y_batch) in enumerate(self._train_data_iter):
            if iter % self._eval_interval == 0 or iter == self._max_iters -1 :
                losses = self._estimate_loss()
                print(f"step {iter}: train loss {losses[0]:.4f}, test loss {losses[1]:.4f}")

            logits, loss = self._model.encoder(x_batch, y_batch)
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()