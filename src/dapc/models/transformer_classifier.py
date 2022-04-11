"""Transformer based classifer implementation."""

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from .abstract_classifier import AbstractClassifier


class TransformerClass(torch.nn.Module):
    """Transformer model."""

    def __init__(self, num_of_labels: Optional[int], model_name: Optional[str]) -> None:
        """Initialize a Transformers+CNN model.

        Args:
            num_of_labels: number of labels.
            model_name: model name or path.
        """
        super(TransformerClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)

        hidden_size = self.l1.config.hidden_size

        self.pre_classifier = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(hidden_size, num_of_labels)  # type: ignore

        torch.nn.init.xavier_uniform(self.pre_classifier.weight)
        torch.nn.init.xavier_uniform(self.classifier.weight)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            input_ids: input ids.
            attention_mask: attention mask.

        Returns:
            logits.
        """
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


class TransformerClassifier(AbstractClassifier):
    def __init__(
        self,
        classes: Optional[int],
        model_name: Optional[str] = "bert-base-uncased",
        max_len: int = 512,
        train_batch_size: int = 16,
        valid_batch_size: int = 64,
        learning_rate: float = 2e-05,
    ):
        """Transformer classifier.

        Args:
            classes: number of classes.
            model_name: model name of path.
            max_len: maximum input length.
            train_batch_size: training batch size.
            valid_batch_size: validation batch size.
            learning_rate: learning rate.
        """
        self.MAX_LEN = max_len
        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VALID_BATCH_SIZE = valid_batch_size
        self.LEARNING_RATE = learning_rate

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.model_max_length < int(1e5):
            self.MAX_LEN = self.tokenizer.model_max_length

        self.model = TransformerClass(classes, model_name)

        if torch.cuda.device_count() > 1:
            logger.info("Utilizing {} GPUs".format(torch.cuda.device_count))
            self.model = nn.DataParallel(self.model)  # type:ignore

        self.model.to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.LEARNING_RATE
        )

    def save(self, path: str) -> None:
        """Save a classifier.

        Args:
            path: path to store the classifier.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load classifier's weights from a checkpoint.

        Args:
            path: path where the checkpoint is located.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def calcuate_accu(self, big_idx: Tensor, targets: List[int]) -> int:
        """Calculate accuracy

        Args:
            big_idx: predicted.
            targets: actual values.

        Returns:
            accuracy.
        """

        return (big_idx == targets).sum().item()  # type:ignore

    def train(self, training_loader: DataLoader, epoch: int) -> None:
        """Perform a training epoch.

        Args:
            training_loader: training set dataloader.
            epoch: current epoch.
        """
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        for _, data in enumerate(training_loader, 0):

            ids = data["input_ids"].to(self.device, dtype=torch.long)
            mask = data["attention_mask"].to(self.device, dtype=torch.long)
            targets = data["labels"].to(self.device, dtype=torch.long)

            outputs = self.model(ids, mask)
            loss = self.loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += self.calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                logger.info(f"Training Loss per 5000 steps: {loss_step}")
                logger.info(f"Training Accuracy per 5000 steps: {accu_step}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info(
            f"The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}"
        )
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        logger.info(f"Training Loss Epoch: {epoch_loss}")
        logger.info(f"Training Accuracy Epoch: {epoch_accu}")

    def fit(self, X: List[str], y: List[int], epochs: int = 3) -> None:
        """Train the classifier.

        Args:
            X: list of input texts.
            y: list of labels.
            epochs: number of epochs.
        """

        train_params = {
            "batch_size": self.TRAIN_BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0,
        }

        train_encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=self.MAX_LEN
        )

        training_set = Dataset(train_encodings, y)

        training_loader = DataLoader(training_set, **train_params)  # type:ignore

        for epoch in range(epochs):
            self.train(training_loader, epoch)

    def valid(self, X: List[str], y: List[int]) -> float:
        """Evaluate an already trained classsifier.

        Args:
            X: list of input texts.
            y: list of labels.

        Returns:
            validation accuracy.
        """

        test_params = {
            "batch_size": self.VALID_BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0,
        }

        test_encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=self.MAX_LEN
        )

        testing_set = Dataset(test_encodings, y)

        testing_loader = DataLoader(testing_set, **test_params)  # type:ignore

        self.model.eval()
        n_correct = 0
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0

        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                targets = data["labels"].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask).squeeze()
                loss = self.loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calcuate_accu(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if _ % 5000 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples
                    logger.info(f"Validation Loss per 100 steps: {loss_step}")
                    logger.info(f"Validation Accuracy per 100 steps: {accu_step}")
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        logger.info(f"Validation Loss Epoch: {epoch_loss}")
        logger.info(f"Validation Accuracy Epoch: {epoch_accu}")

        return epoch_accu

    def predict(self, X: List[str]) -> Tuple[List[int], Optional[Any]]:
        """Predict labels.

        Args:
            X: list of input texts.

        Returns
            list of predicted labels and the respective logits.
        """

        test_params = {
            "batch_size": self.VALID_BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0,
        }

        test_encodings = self.tokenizer(
            X, truncation=True, padding=True, max_length=self.MAX_LEN
        )

        testing_set = Dataset(test_encodings)

        testing_loader = DataLoader(testing_set, **test_params)  # type:ignore

        self.model.eval()

        labels = []
        logits = None

        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):

                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask).squeeze()

                if len(outputs.data.shape) == 1:
                    outputs.data = torch.reshape(outputs.data, (1, -1))

                big_val, big_idx = torch.max(outputs.data, dim=1)

                if logits is None:
                    logits = outputs.data.detach().cpu().numpy()
                else:
                    logits = np.concatenate(
                        (logits, outputs.data.detach().cpu().numpy())
                    )

                labels += [int(x) for x in big_idx]

        return labels, logits


class Dataset(torch.utils.data.Dataset):
    """Dataset class for classifiers."""

    def __init__(self, encodings, labels=None):
        """Initialize a dataset.

        Args
            encodings: list of texts.
            labels:  list of labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """Get an item.

        Args:
            idx: index of the item.

        Returns
            item.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Size of the dataset.

        Returns
            size of the dataset.
        """
        return len(self.encodings["input_ids"])
