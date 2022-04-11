"""Implementation of generic multiclass classifier."""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from .models.abstract_classifier import AbstractClassifier
from .models.adapter_classifier import AdapterClassifier
from .models.cnn_classifier import CNNClassifier
from .models.trans_cnn_classifier import TransCNNClasssifier
from .models.transformer_classifier import TransformerClassifier


class Classifier:
    """Generic multiclass classifier."""

    def __init__(
        self,
        classifier: Union[str, AbstractClassifier],
        num_of_labels: int = None,
        model_name: str = None,
        pretrained_embeddings: str = None,
        corpus: Optional[List[str]] = None,
    ) -> None:
        """Construct a text classifier.

        Args:
            classifier: type of the classifier (transformers, adapters, transformers_cnn and cnn)
                           or an already initialized classifier.
            num_of_labels: number of labels.
            model_name: model name for transformer based classifier.
            pretrained_embeddings: path of pretrained embeddings for CNN classsifier.
            corpus: corpus for embeddings filtering in CNN case.
        """

        self.corpus = corpus
        self.classifier_type = classifier
        self.model_name = model_name
        self.num_of_labels = num_of_labels
        self.pretrained_embeddings = pretrained_embeddings

        self.init_classifier()

    def init_classifier(self) -> None:
        """Initialize a classifier."""

        if (
            not isinstance(self.classifier_type, AbstractClassifier)
            and self.num_of_labels is None
        ):
            raise ValueError(
                "Number of labels should be provided for non-initialized classsifier."
            )

        if isinstance(self.classifier_type, AbstractClassifier):
            self.model = self.classifier_type
        elif self.classifier_type == "transformers":
            self.model = TransformerClassifier(self.num_of_labels, self.model_name)
        elif self.classifier_type == "adapters":
            self.model = AdapterClassifier(self.num_of_labels, self.model_name)
        elif self.classifier_type == "transformers_cnn":
            self.model = TransCNNClasssifier(self.num_of_labels, self.model_name)
        elif self.classifier_type == "cnn":

            if self.pretrained_embeddings is None:
                raise ValueError(
                    "Pretrained embeddings should be provided for CNN classifier."
                )

            self.model = CNNClassifier(
                corpus=self.corpus,
                pretrained_embedding_path=self.pretrained_embeddings,
                num_classes=self.num_of_labels,
            )
        else:
            raise ValueError(f"Classifier {self.classifier_type} not implemented.")

    def save(self, path: str) -> None:
        """Save a classifier.

        Args:
            path: path to store the classifier.
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load classifier's weights from a checkpoint.

        Args:
            path: path where the checkpoint is located.
        """
        self.model.load(path)

    def train(self, X: List[str], y: List[int], epochs: int) -> None:
        """Train the classifier.

        Args:
            X: list of input texts.
            y: list of labels.
            epochs: number of epochs.
        """
        self.model.fit(X, y, epochs)

    def predict(self, X: List[str]) -> Tuple[List[int], Optional[Any]]:
        """Predict labels.

        Args:
            X: list of input texts.

        Returns:
            list of predicted labels and the respected logits.
        """
        return self.model.predict(X)

    def evaluate(self, X: List[str], y: List[int]) -> Dict[str, float]:
        """Evaluate an already trained classsifier.

        Args:
            X: list of input texts.
            y: list of labels.

        Returns:
            dict with evaluation results.
        """

        y_pred, _ = self.predict(X)

        cls = {}
        cls["macro_precision"] = precision_score(y, y_pred, average="macro")
        cls["macro_recall"] = recall_score(y, y_pred, average="macro")
        cls["macro_f1_score"] = f1_score(y, y_pred, average="macro")
        cls["micro_precision"] = precision_score(y, y_pred, average="micro")
        cls["micro_recall"] = recall_score(y, y_pred, average="micro")
        cls["micro_f1_score"] = f1_score(y, y_pred, average="micro")

        return cls

    def cross_validation(
        self,
        X: List[str],
        y: List[int],
        k: int = 20,
        epochs: int = 3,
        save_model_path: str = None,
        save_model_strategy: str = "micro_f1_score",
        shuffle: bool = True,
        random_state: int = 1,
    ) -> Dict[str, List[float]]:
        """Cross validate a classifier and optionally save it.

        Args:
            X: list of input texts.
            y: list of labels.
            k: number of cross validations.
            epochs: number of epochs.
            save_model_path: path where the trained model will be saved. None for no model saving.
            save_model_strategy: save model strategy. `all` to save all the k trained models or a metric.
            shuffle: shuffle the training data.
            random_state: random state.
        """

        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

        cls: Dict[str, List[float]] = {
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1_score": [],
            "micro_precision": [],
            "micro_recall": [],
            "micro_f1_score": [],
        }

        best_save_value = 0.0

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):

            self.init_classifier()

            X_train = [X[idx] for idx in train_index]
            y_train = [y[idx] for idx in train_index]
            X_test = [X[idx] for idx in test_index]
            y_test = [y[idx] for idx in test_index]

            self.train(X_train, y_train, epochs)
            cls_case = self.evaluate(X_test, y_test)

            if save_model_path is not None:

                if save_model_strategy == "all":
                    output_path = os.path.join(save_model_path, str(i))
                    os.makedirs(output_path)
                    self.model.save(output_path)

                else:
                    if cls_case[save_model_strategy] > best_save_value:
                        self.model.save(save_model_path)
                        best_save_value = cls_case[save_model_strategy]

            cls = {x: cls[x] + [cls_case[x]] for x in cls_case}

        for metric in cls:
            logger.info(
                "{}:   min: {:.4f}  max: {:.4f}  mean: {:.4f}  std: {:.4f}".format(
                    metric,
                    min(cls[metric]),
                    max(cls[metric]),
                    np.mean(cls[metric]),
                    np.std(cls[metric]),
                )
            )

        return cls
