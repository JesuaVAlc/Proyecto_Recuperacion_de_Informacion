"""Creación, guardado, carga y consulta del índice FAISS."""
import json
from pathlib import Path

import faiss
import numpy as np


def _normalizar(matriz):
    """Copia float32 contigua con norma L2 = 1 (IP == coseno)."""
    m = np.array(matriz, dtype="float32", order="C", copy=True)
    faiss.normalize_L2(m)
    return m


def crear_indice(embeddings):
    """IndexFlatIP exacto; suficiente para ~20k-60k vectores."""
    m = _normalizar(embeddings)
    indice = faiss.IndexFlatIP(m.shape[1])
    indice.add(m)
    return indice


def guardar(indice, doc_ids, ruta_indice, ruta_ids):
    Path(ruta_indice).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(ruta_indice))
    with open(ruta_ids, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)


def cargar(ruta_indice, ruta_ids):
    indice = faiss.read_index(str(ruta_indice))
    with open(ruta_ids, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    return indice, doc_ids


def buscar(indice, doc_ids, vector_consulta, k):
    q = _normalizar(np.asarray(vector_consulta).reshape(1, -1))
    scores, posiciones = indice.search(q, k)
    return [
        (doc_ids[i], float(s))
        for s, i in zip(scores[0], posiciones[0])
        if i != -1
    ]