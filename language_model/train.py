from dataclasses import dataclass

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets import lm_dataset
from language_model import generation

@dataclass
class TrainArgs:
    n_epochs: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 64
    log_interval: int = 3

class ModelTrainer:
    """
    Class for training language model on given datasets.

    Args:
        model (LanguageModel): The language model which should be trained.
        train_text (list[str]): The training corpus.
        test_text (list[str]): The test corpus.
        args (TrainArgs): The training hyperparameters.
    """
    def __init__(
        self, 
        model: generation.LanguageModel, 
        train_text: list[str], 
        test_text: list[str],
        args: TrainArgs = TrainArgs()
    ):
        self._batch_size = args.batch_size

        self._n_epochs = args.n_epochs
        self._log_interval = args.log_interval

        train_data = lm_dataset.LMDataset(train_text, model.tokenizer, model.context_length)
        self._train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        test_data = lm_dataset.LMDataset(test_text, model.tokenizer, model.context_length)
        self._test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        self._model = model
        self._criterion = nn.functional.cross_entropy
        self._optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.learning_rate)

        self._train_losses: list[float] = []
        self._train_counter: list[int] = []
        self._test_losses: list[float] = []
        self._test_counter: list[int] = [i * len(self._train_loader.dataset) for i in range(self._n_epochs + 1)]


    @torch.no_grad()
    def _test(self) -> None:
        """
        Evaluate the model on the test dataset.
        """
        self._model.encoder.eval()
        test_loss = 0

        for data, targets in self._test_loader:
            output = self._model.encoder(data)

            B, T, C = output.shape

            output = output.view(B * T, C)
            targets = targets.view(B * T)

            test_loss += self._criterion(output, targets).item()

        test_loss /= len(self._test_loader.dataset)
        self._test_losses.append(test_loss)

        print(f'Test loss: {test_loss}')

    def _train_epoch(self, epoch: int) -> None:
        """
        Train the model on the train dataset for one epoch.

        Args:
            epoch (int): Only needed for loss visualization and logging.
        """
        self._model.encoder.train()

        examples_seen = 0

        for batch_idx, (data, targets) in enumerate(self._train_loader):
            self._optimizer.zero_grad()
            output = self._model.encoder(data)

            B, T, C = output.shape

            output = output.view(B * T, C)
            targets = targets.view(B * T)

            loss = self._criterion(output, targets)
            loss.backward()
            self._optimizer.step()

            examples_seen += len(data)

            if batch_idx % self._log_interval == 0:
                self._train_losses.append(loss.item())
                self._train_counter.append((batch_idx * self._batch_size) + (epoch - 1) * len(self._train_loader.dataset))

                print(f'Train epoch {epoch}: [{examples_seen}/{len(self._train_loader.dataset)}] Loss: {loss.item()}')

    def train(self) -> None:
        """
        Train loop for the model. Creates visualization of the loss.
        """
        self._test()
        for epoch in range(1, self._n_epochs + 1):
            self._train_epoch(epoch)
            self._test()

        fig = plt.figure()
        plt.plot(self._train_counter, self._train_losses, color='blue')
        plt.scatter(self._test_counter, self._test_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('Number of training examples seen')
        plt.ylabel('CrossEntropyLoss')
        plt.show()