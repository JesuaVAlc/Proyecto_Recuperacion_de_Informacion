"""Fase offline 1: carga datos, submuestrea y construye el índice invertido."""
import time

import config
from src import data_loader
from src.indice_invertido import IndiceInvertido
from src.preprocesamiento import preprocesar


def main():
    t0 = time.time()
    corpus, consultas, qrels = data_loader.cargar_datos(config.CORPUS_SIZE)
    print(f"Corpus: {len(corpus)} docs | Consultas: {len(consultas)} | "
          f"Pares relevantes: {sum(len(d) for d in qrels.values())}")

    # Artefactos para la web y la evaluación
    data_loader.guardar_json(corpus, config.RUTA_DOC_STORE)
    data_loader.guardar_json(consultas, config.RUTA_CONSULTAS)
    data_loader.guardar_json(qrels, config.RUTA_QRELS)

    # Índice invertido
    docs_tokens = {
        doc_id: preprocesar(data_loader.texto_documento(doc))
        for doc_id, doc in corpus.items()
    }
    indice = IndiceInvertido().construir(docs_tokens)
    indice.guardar(config.RUTA_INDICE_INVERTIDO)

    print(f"Términos únicos: {len(indice.postings)} | "
          f"Longitud promedio: {indice.longitud_promedio:.1f} tokens")
    print(f"Listo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()