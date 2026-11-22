import json
import os
import config
from src import almacen_vectores
from src.data_loader import cargar_json
from src.evaluacion_modelo import analizar_divergencias, evaluar_modelo
from src.indice_invertido import IndiceInvertido
from src.modelos.bm25 import ModeloBM25
from src.modelos.modelo_binario import ModeloJaccard
from src.modelos.semantico import GeneradorEmbeddings, ModeloSemantico
from src.modelos.tfidf_coseno import ModeloTfidfCoseno


def main() -> None:
    print("Cargando artefactos...")
    indice = IndiceInvertido.cargar(config.RUTA_INDICE_INVERTIDO)
    faiss_idx, doc_ids = almacen_vectores.cargar(config.RUTA_FAISS, config.RUTA_DOC_IDS)
    consultas = cargar_json(config.RUTA_CONSULTAS)
    qrels = cargar_json(config.RUTA_QRELS)

    dir_artifacts = os.path.dirname(config.RUTA_CONSULTAS)
    ruta_doc_store = getattr(config, "RUTA_DOC_STORE", os.path.join(dir_artifacts, "doc_store.json"))
    doc_store = cargar_json(ruta_doc_store)

    modelos = [
        ModeloJaccard(indice),
        ModeloTfidfCoseno(indice),
        ModeloBM25(indice),
        ModeloSemantico(GeneradorEmbeddings(), faiss_idx, doc_ids),
    ]

    k = 10
    evaluaciones = {}

    print(f"\nEvaluando {len(modelos)} modelos sobre {len(qrels)} consultas (k={k})...\n")
    print(f"{'Modelo':<25} | {'MAP@10':<8} | {'P@10':<8} | {'R@10':<8} | {'Latencia (ms)':<13}")
    print("-" * 74)

    for modelo in modelos:
        res = evaluar_modelo(modelo, consultas, qrels, k=k)
        evaluaciones[modelo.nombre] = res
        print(
            f"{res['modelo']:<25} | "
            f"{res['MAP']:<8.4f} | "
            f"{res['Precision_promedio']:<8.4f} | "
            f"{res['Recall_promedio']:<8.4f} | "
            f"{res['latencia_promedio_ms']:<13.2f}"
        )

    # 1. Guardar métricas completas (para interfaz web y gráficos)
    ruta_metricas = os.path.join(dir_artifacts, "metricas.json")
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(evaluaciones, f, ensure_ascii=False, indent=2)

    # 2. Guardar casos donde embeddings mejora o empeora vs BM25/TF-IDF (para el informe)
    casos = analizar_divergencias(evaluaciones, consultas, qrels, doc_store, top_casos=5)
    ruta_casos = os.path.join(dir_artifacts, "analisis_comparativo.json")
    with open(ruta_casos, "w", encoding="utf-8") as f:
        json.dump(casos, f, ensure_ascii=False, indent=2)

    print(f"\nMétricas exportadas a: {ruta_metricas}")
    print(f"Casos comparativos exportados a: {ruta_casos}")


if __name__ == "__main__":
    main()  