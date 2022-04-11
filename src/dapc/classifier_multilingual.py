"""Implementation of multiclass classifier including multilingual evaluation capabilities."""
import os
from typing import Dict, List, Union

import numpy as np
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score  # type:ignore
from sklearn.model_selection import StratifiedKFold  # type:ignore

from .classifier import Classifier
from .models.abstract_classifier import AbstractClassifier
from .models.adapter_classifier import AdapterClassifier
from .models.trans_cnn_classifier import TransCNNClasssifier
from .models.transformer_classifier import TransformerClassifier


class ClassifierMultilingual(Classifier):
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

        else:
            raise ValueError(f"Classifier {self.classifier_type} not implemented.")

    def evaluate_multilingual(
        self, X: List[str], y: List[int], lang: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Multilingual evaluation of an already trained classsifier.

        Args:
            X: list of input texts.
            y: list of labels.
            lang: list of languages for each input text.

        Returns:
            dict with evaluation results for each language.
        """

        lang_sets = self.create_lang_sets(lang)

        cls: Dict[str, Dict[str, Union[int, float]]] = {}
        for language in lang_sets:

            X_lang = [X[i] for i in lang_sets[language]]
            y_lang = [y[i] for i in lang_sets[language]]

            if len(X_lang) > 0:
                y_pred, _ = self.predict(X_lang)

                cls[language] = {}
                cls[language]["instances"] = len(y_lang)
                cls[language]["macro_precision"] = precision_score(
                    y_lang, y_pred, average="macro"
                )
                cls[language]["macro_recall"] = recall_score(
                    y_lang, y_pred, average="macro"
                )
                cls[language]["macro_f1_score"] = f1_score(
                    y_lang, y_pred, average="macro"
                )
                cls[language]["micro_precision"] = precision_score(
                    y_lang, y_pred, average="micro"
                )
                cls[language]["micro_recall"] = recall_score(
                    y_lang, y_pred, average="micro"
                )
                cls[language]["micro_f1_score"] = f1_score(
                    y_lang, y_pred, average="micro"
                )

        return cls

    def cross_validation_multilingual(
        self,
        X: List[str],
        y: List[int],
        lang: List[str],
        k: int = 20,
        epochs: int = 3,
        save_model_path: str = None,
        save_model_strategy: str = "micro_f1_score",
        shuffle: bool = True,
        random_state: int = 1,
    ) -> Dict[str, Dict[str, List[float]]]:
        """Multilingual cross validation of a classifier.

        Args:
            X: list of input texts.
            y: list of labels.
            lang: list of languages for each input text.
            k: number of cross validations.
            epochs: number of epochs.
            save_model_path: path where the trained model will be saved. None for no model saving.
            save_model_strategy: save model strategy. `all` to save all the k trained models or a metric.
            shuffle: shuffle the training data.
            random_state: random state.
        """

        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)

        lang_sets = self.create_lang_sets(lang)

        cls: Dict[str, Dict[str, List[float]]] = {
            lang: {
                "macro_precision": [],
                "macro_recall": [],
                "macro_f1_score": [],
                "micro_precision": [],
                "micro_recall": [],
                "micro_f1_score": [],
            }
            for lang in lang_sets
        }

        best_save_value = 0.0

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):

            self.init_classifier()

            X_train = [X[idx] for idx in train_index]
            y_train = [y[idx] for idx in train_index]
            X_test = [X[idx] for idx in test_index]
            y_test = [y[idx] for idx in test_index]
            lang_test = [lang[idx] for idx in test_index]

            self.train(X_train, y_train, epochs)
            cls_case = self.evaluate_multilingual(X_test, y_test, lang_test)

            if save_model_path is not None:

                if save_model_strategy == "all":
                    output_path = os.path.join(save_model_path, str(i))
                    os.makedirs(output_path)
                    self.model.save(output_path)

                else:
                    if cls_case["all"][save_model_strategy] > best_save_value:
                        self.model.save(save_model_path)
                        best_save_value = cls_case["all"][save_model_strategy]

            for language in lang_sets:
                if language in cls_case:
                    cls[language] = {
                        x: cls[language][x] + [cls_case[language][x]]
                        for x in cls_case[language]
                        if x != "instances"
                    }

        for language in lang_sets:
            for metric in cls[language]:
                if len(cls[language][metric]):
                    logger.info(
                        "{} - {}:   min: {:.4f}  max: {:.4f}  mean: {:.4f}  std: {:.4f}".format(
                            language,
                            metric,
                            min(cls[language][metric]),
                            max(cls[language][metric]),
                            np.mean(cls[language][metric]),
                            np.std(cls[language][metric]),
                        )
                    )

        return cls

    def create_lang_sets(self, lang: List[str]) -> Dict[str, List[int]]:
        """Sort text indices based on their language.

        Args:
            lang: list of languages for each input text.

        Returns:
            dictionary with all the indices for each language.
        """

        languages = set(lang)

        language_sets = {}

        for language in languages:
            language_sets[language] = [i for i, la in enumerate(lang) if la == language]

        language_sets["all"] = list(range(len(lang)))
        language_sets["non-en"] = [i for i, la in enumerate(lang) if la != "en"]

        return language_sets
