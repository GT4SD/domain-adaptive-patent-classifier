"""Labels converter."""
from typing import List, Union


class LabelsConverter:
    """Labels convertion helper class."""

    def __init__(self, labels: List[Union[str, int]]) -> None:
        """Create an LabelsConverter instance by providing a list of labels.

        Args:
            labels: list of labels.
        """
        self.classes = list(set(labels))
        self.classes.sort()

        self.str2id = {x: i for i, x in enumerate(self.classes)}
        self.id2str = {i: x for i, x in enumerate(self.classes)}

    def encode_labels(self, labels: List[str]) -> List[int]:
        """Convert string labels to int.

        Args:
            labels: list of string labels.

        Returns:
            list of int labels.
        """
        return [self.str2id[x] for x in labels]

    def decode_labels(self, labels: List[int]) -> List[Union[str, int]]:
        """Convert int labels to string.

        Args:
            labels: list of int labels.

        Returns:
            list of string labels.
        """
        return [self.id2str[x] for x in labels]
