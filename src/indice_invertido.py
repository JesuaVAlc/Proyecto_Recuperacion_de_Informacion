"""Índice invertido propio: término -> {doc_id: tf}."""
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path


class IndiceInvertido:
    def __init__(self):
        self.postings = {}          # término -> {doc_id: tf}
        self.longitudes = {}        # doc_id -> nº de tokens (para BM25)
        self.n_unicos = {}          # doc_id -> nº de términos distintos (para Jaccard)
        self.norma_tfidf = {}       # doc_id -> norma del vector TF-IDF
        self.n_docs = 0
        self.longitud_promedio = 0.0

    # ---------- Construcción ----------
    def construir(self, docs_tokens):
        """docs_tokens: {doc_id: [tokens ya preprocesados]}"""
        postings = defaultdict(dict)
        for doc_id, tokens in docs_tokens.items():
            frecuencias = Counter(tokens)
            self.longitudes[doc_id] = len(tokens)
            self.n_unicos[doc_id] = len(frecuencias)
            for termino, tf in frecuencias.items():
                postings[termino][doc_id] = tf

        self.postings = dict(postings)
        self.n_docs = len(self.longitudes)
        total = sum(self.longitudes.values())
        self.longitud_promedio = total / self.n_docs if self.n_docs else 0.0
        self._calcular_normas_tfidf()
        return self

    def _calcular_normas_tfidf(self):
        sumas = defaultdict(float)
        for termino, docs in self.postings.items():
            idf = self.idf_tfidf(termino)
            for doc_id, tf in docs.items():
                sumas[doc_id] += (self.peso_tf(tf) * idf) ** 2
        self.norma_tfidf = {d: math.sqrt(s) for d, s in sumas.items()}

    # ---------- Estadísticas ----------
    def df(self, termino):
        return len(self.postings.get(termino, ()))

    @staticmethod
    def peso_tf(tf):
        return 1.0 + math.log(tf) if tf > 0 else 0.0

    def idf_tfidf(self, termino):
        df = self.df(termino)
        return math.log(self.n_docs / df) if df else 0.0

    def idf_bm25(self, termino):
        df = self.df(termino)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)) if df else 0.0

    def documentos_con(self, termino):
        return self.postings.get(termino, {})

    # ---------- Persistencia ----------
    def guardar(self, ruta):
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def cargar(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)