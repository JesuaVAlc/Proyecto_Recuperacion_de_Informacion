"""Modelo probabilístico BM25 (Okapi)."""
from collections import defaultdict

import config
from src.modelos.base import ModeloRecuperacion, top_k
from src.preprocesamiento import preprocesar


class ModeloBM25(ModeloRecuperacion):
    nombre = "Probabilístico (BM25)"

    def __init__(self, indice, k1=config.BM25_K1, b=config.BM25_B):
        self.indice = indice
        self.k1 = k1
        self.b = b

    def buscar(self, consulta, k=10):
        terminos_q = set(preprocesar(consulta))
        if not terminos_q:
            return []

        avgdl = self.indice.longitud_promedio
        scores = defaultdict(float)
        for t in terminos_q:
            idf = self.indice.idf_bm25(t)
            if idf == 0:
                continue
            for doc_id, tf in self.indice.documentos_con(t).items():
                dl = self.indice.longitudes[doc_id]
                denominador = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / denominador
        return top_k(scores, k)