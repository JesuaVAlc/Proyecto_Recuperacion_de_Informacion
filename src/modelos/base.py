"""Interfaz común: todos los modelos exponen buscar(consulta, k)."""
import heapq
from abc import ABC, abstractmethod


class ModeloRecuperacion(ABC):
    nombre = "base"

    @abstractmethod
    def buscar(self, consulta, k=10):
        """Retorna [(doc_id, score)] ordenado de mayor a menor score."""


def top_k(scores, k):
    """Top-k de {doc_id: score}; desempata por doc_id para ser determinista."""
    return heapq.nsmallest(k, scores.items(), key=lambda x: (-x[1], x[0]))