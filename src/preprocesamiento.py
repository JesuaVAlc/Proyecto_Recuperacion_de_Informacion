"""Tokenización, normalización, stopwords y stemming."""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_PATRON_TOKEN = re.compile(r"[a-z0-9]+")
_stemmer = PorterStemmer()


def _asegurar_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOPWORDS = _asegurar_stopwords()


@lru_cache(maxsize=200_000)
def _stem(token):
    return _stemmer.stem(token)


def tokenizar(texto):
    """Minúsculas + extracción de tokens alfanuméricos."""
    return _PATRON_TOKEN.findall(texto.lower())


def remover_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocesar(texto):
    """Pipeline completo: tokenizar -> stopwords -> stemming."""
    return [_stem(t) for t in remover_stopwords(tokenizar(texto))]