"""CNN classifer implementation."""

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from nltk import word_tokenize  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from .abstract_classifier import AbstractClassifier


class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for text Classification."""

    def __init__(
        self,
        pretrained_embedding: Tensor,
        freeze_embedding: bool = False,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: List[int] = [100, 100, 100],
        num_classes: Optional[int] = 2,
        dropout: float = 0.2,
    ) -> None:
        """Constructor of CNN model.

        Args:
            pretrained_embedding: pretrained embeddings.
            freeze_embedding: set to False to fine-tune pretraiend vectors.
            filter_sizes: list of filter sizes.
            num_filters: list of number of filters.
            num_classes: number of classes.
            dropout: dropout rate.
        """

        super(CNN_NLP, self).__init__()

        self.vocab_size, self.embed_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embedding, freeze=freeze_embedding
        )

        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=num_filters[i],
                    kernel_size=filter_sizes[i],
                )
                for i in range(len(filter_sizes))
            ]
        )

        self.fc = nn.Linear(np.sum(num_filters), num_classes)  # type: ignore
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Perform a forward pass.

        Args:
            input_ids: a tensor of token ids.

        Returns:
            output logits.
        """

        x_embed = self.embedding(input_ids).float()

        x_reshaped = x_embed.permute(0, 2, 1)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]

        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        logits = self.fc(self.dropout(x_fc))

        return logits


class CNNClassifier(AbstractClassifier):
    """ CNN classifier."""

    def __init__(
        self,
        corpus: Optional[List[str]],
        pretrained_embedding_path: str,
        freeze_embedding: bool = False,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: List[int] = [100, 100, 100],
        num_classes: Optional[int] = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        max_len: int = 512,
    ) -> None:
        """Initialize a CNN classifier.

        Args:
            corpus: corpus to determine model's vocabulary.
            pretrained_embedding_path: path where the pretrained embeddings are located.
            freeze_embedding: set to False to fine-tune pretraiend vectors.
            filter_sizes: list of filter sizes.
            num_filters: list of number of filters.
            num_classes: number of classes.
            dropout: dropout.
            learning_rate: learning rate.
            batch_size: batch size.
            max_len: maximum input length.
        """

        assert len(filter_sizes) == len(
            num_filters
        ), "filter_sizes and \
        num_filters need to be of the same length."

        self.batch_size = batch_size
        self.max_len = max_len

        pretrained_embedding, self.word2idx, _ = self.load_pretrained_vectors(
            corpus, pretrained_embedding_path
        )

        self.model = CNN_NLP(
            pretrained_embedding=pretrained_embedding,
            freeze_embedding=freeze_embedding,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.model.to(self.device)

        self.optimizer = optim.Adadelta(
            self.model.parameters(), lr=learning_rate, rho=0.95
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def tokenize(self, texts: List[str]) -> Tuple[List[List[str]], Dict[str, int], int]:
        """Tokenize texts.

        Args:
            texts: list of text data.

        Returns:
            tokenized_texts: list of list of tokens.
            word2idx: vocabulary built from the corpus.
            max_len: maximum sentence length.
        """

        max_len = 0
        tokenized_texts = []
        word2idx = {}

        word2idx["<pad>"] = 0
        word2idx["<unk>"] = 1

        idx = 2
        for sent in texts:
            tokenized_sent = word_tokenize(sent)

            tokenized_texts.append(tokenized_sent)

            for token in tokenized_sent:
                if token not in word2idx:
                    word2idx[token] = idx
                    idx += 1

            max_len = max(max_len, len(tokenized_sent))

        return tokenized_texts, word2idx, max_len

    def encode(
        self, tokenized_texts: List[List[str]], word2idx: Dict[str, int], max_len: int
    ) -> Any:
        """Pad each sentence to the maximum sentence length and encode tokens to
        their index in the vocabulary.

        Args:
            tokenized_texts: list of list of tokens.
            word2idx: vocabulary built from the corpus.
            max_len: maximum sentence length.

        Returns:
            array of token indexes in the vocabulary.
        """

        input_ids = []

        for tokenized_sent in tokenized_texts:
            tokenized_sent = tokenized_sent[:max_len]
            tokenized_sent += ["<pad>"] * (max_len - len(tokenized_sent))

            input_id = [
                word2idx.get(token, word2idx["<unk>"]) for token in tokenized_sent
            ]
            input_ids.append(input_id)

        return np.array(input_ids)

    def fit(self, X: List[str], y: List[int], epochs: int = 40) -> None:
        """Train the classifier.

        Args:
            X: list of input texts.
            y: list of labels.
            epochs: number of epochs.
        """

        tokenized_texts, word2idx, _ = self.tokenize(X)
        input_ids = self.encode(tokenized_texts, word2idx, self.max_len)

        train_dataloader = self.data_loader(input_ids, y, batch_size=self.batch_size)

        for epoch_i in range(epochs):

            t0_epoch = time.time()
            total_loss = 0

            self.model.train()

            for step, batch in enumerate(train_dataloader):

                b_input_ids, b_labels = tuple(t.to(self.device) for t in batch)

                self.model.zero_grad()

                logits = self.model(b_input_ids)

                loss = self.loss_fn(logits, b_labels)
                total_loss += loss.item()

                loss.backward()

                self.optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)

            _, train_accuracy = self.valid(val_dataloader=train_dataloader)

            time_elapsed = time.time() - t0_epoch
            logger.info(
                "Epoch: {} Average train loss: {} Train accuracy: {} Time elapsed: {}".format(
                    epoch_i, avg_train_loss, train_accuracy, time_elapsed
                )
            )

    def load_vectors(self, word2idx: Dict[str, int], fname: str) -> Any:
        """Load pretrained vectors and create embedding layers.

        Args:
            word2idx: vocabulary built from the corpus.
            fname: path to pretrained vector file.

        Returns:
            embedding matrix.
        """

        logger.info("Loading pretrained vectors...")

        with open(fname, "rb") as f:
            embeddings_dict = pickle.load(f)

        d = embeddings_dict[list(embeddings_dict.keys())[0]].shape[0]

        embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
        embeddings[word2idx["<pad>"]] = np.zeros((d,))
        embeddings[word2idx["<unk>"]] = np.zeros((d,))

        count = 0
        for word in tqdm(embeddings_dict):
            if word in word2idx:
                count += 1
                embeddings[word2idx[word]] = np.array(
                    embeddings_dict[word], dtype=np.float32
                )

        return embeddings

    def load_pretrained_vectors(
        self, corpus: Optional[List[str]], pretrained_embedding_path: str
    ) -> Tuple[Tensor, Dict[str, int], int]:
        """Load pretrained embeddng vectors.

        Args:
            corpus: corpus to determine the vocabulary.
            pretrained_embedding_path: path where the pretrained embeddings are located.

        Returns:
            embeddings: embedding matrix.
            word2idx: vocabulary built from the corpus.
            max_len: ,aximum sentence length.
        """
        tokenized_texts, word2idx, max_len = self.tokenize(corpus)  # type: ignore

        embeddings_ = self.load_vectors(word2idx, pretrained_embedding_path)
        embeddings: Tensor = torch.tensor(embeddings_)  # type: ignore

        return embeddings, word2idx, max_len

    def data_loader(
        self,
        inputs: Any,
        labels: List[int] = None,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        """Convert train and validation sets to torch.Tensors and load them to
        DataLoader.

        Args:
            inputs: input ids.
            labels: list of labels.
            batch_size: batch size.
            shuffle: shuffle instances.

        Returns:
            instances' dataloader.
        """

        train_inputs = torch.tensor(inputs)

        if labels is not None:
            labels_ = torch.tensor(labels)
            data = TensorDataset(train_inputs, labels_)
        else:
            data = TensorDataset(train_inputs)

        sampler = None
        if shuffle:
            sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def valid(self, X=None, y=None, val_dataloader=None) -> Tuple[float, float]:
        """Validate the classifier be either provide a dataloader or texts and their labels.

        Args:
            X: list of texts.
            y: list of labels.
            val_dataloader: validation's dataloader.

        Returns:
            validation's loss and accuracy.
        """

        if val_dataloader is None:
            tokenized_texts, _, _ = self.tokenize(X)
            input_ids = self.encode(tokenized_texts, self.word2idx, self.max_len)

            val_dataloader = self.data_loader(input_ids, y, batch_size=self.batch_size)

        self.model.eval()

        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            b_input_ids, b_labels = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                logits = self.model(b_input_ids)

            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss_mean = np.mean(val_loss)
        val_accuracy_mean = np.mean(val_accuracy)

        return val_loss_mean, val_accuracy_mean

    def predict(self, texts):
        """Predict labels.

        Args:
            X: list of input texts.

        Returns:
            list of predicted labels and their logits.
        """

        tokenized_texts, _, _ = self.tokenize(texts)
        input_ids = self.encode(tokenized_texts, self.word2idx, self.max_len)

        dataloader = self.data_loader(
            input_ids, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()

        labels = []
        logits = None

        for batch in dataloader:

            b_input_ids = batch[0].to(self.device)

            with torch.no_grad():
                batch_logits = self.model(b_input_ids)

            big_val, big_idx = torch.max(batch_logits, dim=1)

            if logits is None:
                logits = batch_logits.detach().cpu().numpy()
            else:
                logits = np.concatenate((logits, batch_logits.detach().cpu().numpy()))

            labels += [int(x) for x in big_idx]

        return labels, logits
