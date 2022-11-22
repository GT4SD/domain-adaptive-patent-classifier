import pickle

import click
import pandas as pd

from dapc.classifier_multilingual import ClassifierMultilingual
from dapc.labels_converter import LabelsConverter
from dapc.languages_detector import LanguagesDetector


@click.command()
@click.command()
@click.option(
    "--datapath",
    type=str,
    required=True,
    help="Enter the path for the training dataset.",
)
@click.option(
    "--model_type",
    type=str,
    required=True,
    help="Enter the model type (transformers, adapters or cnn).",
)
@click.option(
    "--model_name", type=str, required=True, help="Enter the model name or path."
)
@click.option("--epochs", type=int, required=True, help="Enter the number of epochs.")
@click.option(
    "--k",
    type=int,
    default=10,
    help="Enter the number of k folds for cross validation.",
)
def main(datapath, model_type, model_name, epochs, k):
    if datapath.endswith(".pl"):
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        texts = data["texts"]
        labels = data["labels"]

    elif datapath.endswith(".xlsx"):
        data = pd.read_excel(datapath)

        texts = data["texts"].to_list()
        labels = data["labels"].to_list()

    else:
        raise ValueError("Datapath should be a pickle or excel file.")

    lb = LabelsConverter(labels)

    labels_int = lb.encode_labels(labels)

    transformer_classifier_multi = ClassifierMultilingual(
        model_type, model_name=model_name, num_of_labels=len(lb.classes)
    )

    lan_detector = LanguagesDetector()
    langs = lan_detector.infer_languages(texts)

    transformer_classifier_multi.cross_validation_multilingual(
        texts, labels_int, langs, k=k, epochs=epochs
    )


if __name__ == "__main__":
    main()
