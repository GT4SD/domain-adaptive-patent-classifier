"""Languages detection helper."""
from typing import List

import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy_langdetect import LanguageDetector  # type: ignore


def get_lang_detector(nlp, name):
    return LanguageDetector()


class LanguagesDetector:
    """Languages detector."""

    def __init__(self) -> None:
        """Create a languages detector class."""

        self.nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        self.nlp.add_pipe("language_detector", last=True)

    def infer_languages(self, X: List[str]) -> List[str]:
        """Infer languages

        Args:
            X: list of texts.

        Returns:
            list of the respective languages.
        """

        langs = []

        for x in X:
            doc = self.nlp(x)
            lang = doc._.language["language"]
            langs.append(lang)

        return langs
