"""Fase offline 2: embeddings de los documentos + índice FAISS."""
import time

import config
from src import almacen_vectores, data_loader
from src.modelos.semantico import GeneradorEmbeddings


def main():
    t0 = time.time()
    corpus = data_loader.cargar_json(config.RUTA_DOC_STORE)
    doc_ids = list(corpus.keys())
    textos = [data_loader.texto_documento(corpus[d]) for d in doc_ids]

    generador = GeneradorEmbeddings()
    embeddings = generador.embeber_documentos(textos)
    print(f"Matriz de embeddings: {embeddings.shape}")

    indice = almacen_vectores.crear_indice(embeddings)
    almacen_vectores.guardar(indice, doc_ids, config.RUTA_FAISS, config.RUTA_DOC_IDS)
    print(f"Vectores en FAISS: {indice.ntotal} | Listo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()