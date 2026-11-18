"""Recuperación semántica: embeddings (fastembed/ONNX) + FAISS."""
import numpy as np
from tqdm import tqdm

import config
from src import almacen_vectores
from src.modelos.base import ModeloRecuperacion


class GeneradorEmbeddings:
    """Mismo modelo para documentos (offline) y consultas (online)."""

    def __init__(self, nombre_modelo=config.MODELO_EMBEDDINGS):
        from fastembed import TextEmbedding  # import diferido: solo si se usa
        config.DIR_CACHE_MODELO.mkdir(parents=True, exist_ok=True)
        self.modelo = TextEmbedding(
            model_name=nombre_modelo,
            cache_dir=str(config.DIR_CACHE_MODELO),
        )

    def embeber_documentos(self, textos, tamano_lote=config.TAMANO_LOTE_EMBEDDINGS):
        generador = self.modelo.embed(textos, batch_size=tamano_lote)
        vectores = list(tqdm(generador, total=len(textos), desc="Embeddings"))
        return np.array(vectores, dtype="float32")

    def embeber_consulta(self, consulta):
        return np.array(next(iter(self.modelo.embed([consulta]))), dtype="float32")


class ModeloSemantico(ModeloRecuperacion):
    nombre = "Vectorial (Embeddings)"

    def __init__(self, generador, indice_faiss, doc_ids):
        self.generador = generador
        self.indice_faiss = indice_faiss
        self.doc_ids = doc_ids

    def buscar(self, consulta, k=10):
        if not consulta.strip():
            return []
        vector = self.generador.embeber_consulta(consulta)
        return almacen_vectores.buscar(self.indice_faiss, self.doc_ids, vector, k)