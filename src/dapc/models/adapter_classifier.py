"""Adapter classifier implementation."""
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AdapterConfig, AutoModelWithHeads, AutoTokenizer

from .transformer_classifier import Dataset, TransformerClassifier


class AdapterClassifier(TransformerClassifier):
    def __init__(
        self,
        classes: Optional[int],
        model_name: Optional[str] = "bert-base-uncased",
        task_name: str = "classification",
        max_len: int = 512,
        train_batch_size: int = 16,
        valid_batch_size: int = 64,
        learning_rate: float = 0.0001,
    ) -> None:
        """Initialize an adapter classifier

        Args:
            classes: number of classes.
            model_name: model name or path.
            task_name: task name.
            max_len: maximum length.
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

        self.model = AutoModelWithHeads.from_pretrained(model_name)

        self.task_name = task_name

        self.model.add_classification_head(  # type:ignore
            self.task_name, num_labels=classes
        )

        adapter_config = AdapterConfig.load("pfeiffer")

        self.model.add_adapter(self.task_name, config=adapter_config)  # type:ignore

        self.model.train_adapter([self.task_name])  # type:ignore
        self.model.set_active_adapters([self.task_name])  # type:ignore

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
        self.model.save_adapter(path, self.task_name)  # type:ignore

    def load(self, path: str) -> None:
        """Load classifier's weights from a checkpoint.

        Args:
            path: path where the checkpoint is located.
        """
        self.model.set_active_adapters(None)  # type:ignore
        self.model.delete_adapter(self.task_name)  # type:ignore
        self.model.load_adapter(path, load_as=self.task_name)  # type:ignore
        self.model.set_active_adapters([self.task_name])  # type:ignore
        self.model.to(self.device)

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
            loss = self.loss_function(outputs.logits, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.logits, dim=1)
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

    def predict(self, X: List[str]) -> Tuple[List[int], Optional[Any]]:
        """Predict labels.

        Args:
            X: list of input texts.

        Returns:
            list of predicted labels and their logits.
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
                outputs = self.model(ids, mask).logits.squeeze()

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
