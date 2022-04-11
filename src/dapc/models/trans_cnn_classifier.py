"""Transformer and CNN based classifer implementation."""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .transformer_classifier import TransformerClassifier


class CNNTransformerClass(torch.nn.Module):
    """Transformers+CNN model."""

    def __init__(
        self,
        num_of_labels: Optional[int],
        model_name: Optional[str],
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: List[int] = [200, 200, 200],
        dropout: float = 0.5,
    ) -> None:
        """Initialize a Transformers+CNN model.

        Args:
            num_of_labels: number of labels.
            model_name: model name or path.
            filter_sizes: list of filter sizes.
            num_filters: list of number of filters.
            dropout: dropout.
        """

        super(CNNTransformerClass, self).__init__()

        self.model = AutoModel.from_pretrained(
            model_name, return_dict=True, output_hidden_states=True
        )

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_value = dropout
        self.num_of_labels = num_of_labels

        hidden_size = self.model.config.hidden_size

        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=self.num_filters[i],
                    kernel_size=self.filter_sizes[i],
                )
                for i in range(len(self.filter_sizes))
            ]
        )

        self.fc = nn.Linear(np.sum(self.num_filters), self.num_of_labels)  # type:ignore
        self.dropout = nn.Dropout(p=self.dropout_value)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            input_ids: input ids.
            attention_mask: attention mask.

        Returns:
            logits.
        """
        output_1 = self.model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]

        x_reshaped = hidden_state.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc(self.dropout(x_fc))

        return logits


class TransCNNClasssifier(TransformerClassifier):
    def __init__(
        self,
        classes: Optional[int],
        model_name: Optional[str] = "bert-base-uncased",
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: List[int] = [200, 200, 200],
        dropout: float = 0.5,
        max_len: int = 512,
        train_batch_size: int = 16,
        valid_batch_size: int = 64,
        learning_rate: float = 2e-05,
    ) -> None:
        """Transformer+cnn classifier.

        Args:
            classes: number of classes.
            model_name: model name of path.
            filter_sizes: list of filter sizes.
            num_filters: list of number of filters.
            dropout: dropout.
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="[PAD]")

        if self.tokenizer.model_max_length < int(1e5):
            self.MAX_LEN = self.tokenizer.model_max_length

        self.model = CNNTransformerClass(  # type:ignore
            classes, model_name, filter_sizes, num_filters, dropout
        )

        self.model.model.resize_token_embeddings(len(self.tokenizer))  # type:ignore

        if torch.cuda.device_count() > 1:
            logger.info("Utilizing {} GPUs".format(torch.cuda.device_count))
            self.model = nn.DataParallel(self.model)  # type:ignore

        self.model.to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.LEARNING_RATE
        )
