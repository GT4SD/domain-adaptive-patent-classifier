"""Abstract classifer implementation."""
from typing import Any, List, Optional, Tuple


class AbstractClassifier:
    """Abstract classifier"""

    def save(self, path: str) -> None:
        """Save a classifier.

        Args:
            path: path to store the classifier.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load classifier's weights from a checkpoint.

        Args:
            path: path where the checkpoint is located.
        """
        raise NotImplementedError

    def fit(self, X: List[str], y: List[int], epochs: int) -> None:
        """Train the classifier.

        Args:
            X: list of input texts.
            y: list of labels.
            epochs: number of epochs.
        """
        raise NotImplementedError

    def predict(self, X: List[str]) -> Tuple[List[int], Optional[Any]]:
        """Predict labels.

        Args:
            X: list of input texts.

        Returns:
            list of predicted labels.
        """
        raise NotImplementedError
