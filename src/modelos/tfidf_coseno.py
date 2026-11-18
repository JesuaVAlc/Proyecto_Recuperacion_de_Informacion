"""Modelo vectorial: TF-IDF (tf logarítmico, idf = log N/df) + coseno."""
import math
from collections import Counter, defaultdict

from src.modelos.base import ModeloRecuperacion, top_k
from src.preprocesamiento import preprocesar


class ModeloTfidfCoseno(ModeloRecuperacion):
    nombre = "Vectorial (TF-IDF)"

    def __init__(self, indice):
        self.indice = indice

    def _pesos_consulta(self, consulta):
        pesos = {}
        for t, f in Counter(preprocesar(consulta)).items():
            idf = self.indice.idf_tfidf(t)
            if idf > 0:
                pesos[t] = self.indice.peso_tf(f) * idf
        return pesos

    def buscar(self, consulta, k=10):
        pesos_q = self._pesos_consulta(consulta)
        norma_q = math.sqrt(sum(w * w for w in pesos_q.values()))
        if norma_q == 0:
            return []

        producto = defaultdict(float)
        for t, wq in pesos_q.items():
            idf = self.indice.idf_tfidf(t)
            for doc_id, tf in self.indice.documentos_con(t).items():
                producto[doc_id] += wq * self.indice.peso_tf(tf) * idf

        scores = {}
        for doc_id, p in producto.items():
            norma_d = self.indice.norma_tfidf.get(doc_id, 0.0)
            if norma_d > 0:
                scores[doc_id] = p / (norma_q * norma_d)
        return top_k(scores, k)