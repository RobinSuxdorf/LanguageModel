from typing import Any
from dataclasses import dataclass

import json

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets import lm_dataset
from language_model import generation
from tokenizers import special_tokens

@dataclass
class TrainArgs:
    n_epochs: int = 3
    learning_rate: float = 3e-4
    batch_size: int = 32
    log_interval: int = 3
    train_test_split_ratio: float = 0.9

class ModelTrainer:
    """
    Class for training language model on given datasets.

    Args:
        model (LanguageModel): The language model which should be trained.
        data (list[dict[str, Any]]): The training/test corpus for the language model.
        args (TrainArgs): The training hyperparameters.
    """
    def __init__(
        self, 
        model: generation.LanguageModel, 
        data: list[dict[str, Any]],
        args: TrainArgs = TrainArgs()
    ) -> None:
        self._batch_size = args.batch_size

        self._n_epochs = args.n_epochs
        self._log_interval = args.log_interval

        corpus = self._prepate_data(data, model)

        split_value: int = int(args.train_test_split_ratio * len(corpus))
        train_data: list[dict[str, Any]] = corpus[:split_value]
        test_data: list[dict[str, Any]] = corpus[split_value:]

        train_data = lm_dataset.LMDataset(train_data)
        self._train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        test_data = lm_dataset.LMDataset(test_data)
        self._test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

        self._model = model
        self._criterion = nn.functional.cross_entropy
        self._optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.learning_rate)

        self._train_losses: list[float] = []
        self._train_counter: list[int] = []
        self._test_losses: list[float] = []
        self._test_counter: list[int] = [i * len(self._train_loader.dataset) for i in range(self._n_epochs + 1)]

    def _prepate_data(self, data: list[dict[str, Any]], model: generation.LanguageModel) -> list[dict[str, Any]]:
        """
        Fuction for data preparation e.g. filter out texts longer than context length and generating encodig.

        Args:
            data (lists[dict[str, Any]]): The data which which will be prepared.
            model (generation.LanguageModel): The model for which the data will be prepared.

        Returns:
            list[dict[str, Any]]: The encoded data set.
        """
        corpus: list[dict[str, Any]] = []

        if 'encoded'  in data[0]:
            corpus = [text for text in data if len(text['encoded']) == model.context_length + 1]
        else:
            for text in data:
                text['encoded'] = model.tokenizer.encode(text['summary'])

            corpus =  [text for text in data if len(text['encoded']) < model.context_length]

            for text in corpus:
                encoded_text: list[int] = text['encoded']
                padded_encoded_text = [0 for _ in range(model.context_length + 1)]

                padded_encoded_text[1:len(encoded_text) + 1] = encoded_text

                padded_encoded_text[0] = model.tokenizer.stoi[special_tokens.SpecialTokens.SOS]
                padded_encoded_text[len(encoded_text) + 1] = model.tokenizer.stoi[special_tokens.SpecialTokens.EOS]

                pad_tokens: list[int] = [model.tokenizer.stoi[special_tokens.SpecialTokens.PAD] for _ in range(model.context_length - len(encoded_text) - 1)]
                padded_encoded_text[len(encoded_text) + 2:model.context_length + 1] = pad_tokens

                text['encoded'] = padded_encoded_text

            corpus_json = json.dumps(corpus)

            with open(f'./datasets/data_{model.context_length}_tokens.json', 'w') as f:
                f.write(corpus_json)

        return corpus

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

            if batch_idx % self._log_interval == 0 or examples_seen == len(self._train_loader.dataset):
                self._train_losses.append(loss.item())
                self._train_counter.append((batch_idx * self._batch_size) + (epoch - 1) * len(self._train_loader.dataset))

                print(f'Train epoch [{epoch}/{self._n_epochs}]: [{examples_seen}/{len(self._train_loader.dataset)}] Loss: {loss.item()}')

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