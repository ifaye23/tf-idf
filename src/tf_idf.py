"""TF-IDF module."""

import math

from textblob import TextBlob


def tf(word: str, blob: TextBlob) -> float:
    """Calculate the term frequency of a word in a TextBlob.

    :param word: a word
    :param blob: a TextBlob object
    :return: the term frequency of the word in the blob
    """
    return blob.words.count(word) / len(blob.words)  # type: ignore


def n_containing(word: str, bloblist: list[TextBlob]) -> int:
    """Count the number of TextBlob objects that contain a word.

    :param word: a word
    :param bloblist: a list of TextBlob objects
    :return: the number of blobs that contain the word
    """
    return sum(1 for blob in bloblist if word in blob.words)  # type: ignore


def idf(word: str, bloblist: list[TextBlob]) -> float:
    """Calculate the inverse document frequency of a word in a list of TextBlob objects.

    :param word: a word
    :param bloblist: a list of TextBlob objects
    :return: the inverse document frequency of the word in the bloblist
    """
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))  # type: ignore


def tfidf(word: str, blob: TextBlob, bloblist: list[TextBlob]) -> float:
    """Calculate the TF-IDF score of a word in a TextBlob.

    :param word: a word
    :param blob: a TextBlob object
    :param bloblist: a list of TextBlob objects
    :return: the TF-IDF score of the word in the blob
    """
    return tf(word, blob) * idf(word, bloblist)  # type: ignore
