"""Modelo binario: similitud de Jaccard entre conjuntos de términos."""
from collections import defaultdict

from src.modelos.base import ModeloRecuperacion, top_k
from src.preprocesamiento import preprocesar


class ModeloJaccard(ModeloRecuperacion):
    nombre = "Binario (Jaccard)"

    def __init__(self, indice):
        self.indice = indice

    def buscar(self, consulta, k=10):
        terminos_q = set(preprocesar(consulta))
        if not terminos_q:
            return []

        # |Q ∩ D| para cada documento candidato, usando solo las postings
        interseccion = defaultdict(int)
        for t in terminos_q:
            for doc_id in self.indice.documentos_con(t):
                interseccion[doc_id] += 1

        n_q = len(terminos_q)
        scores = {
            d: i / (n_q + self.indice.n_unicos[d] - i)   # |Q∩D| / |Q∪D|
            for d, i in interseccion.items()
        }
        return top_k(scores, k)