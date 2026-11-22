import time
from typing import Any


def precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """Proporción de documentos relevantes dentro del top-k recuperado."""
    if k <= 0:
        return 0.0
    top_k = recuperados[:k]
    aciertos = sum(1 for doc_id in top_k if doc_id in relevantes)
    return aciertos / k


def recall_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """Proporción del total de documentos relevantes que fueron recuperados en el top-k."""
    if not relevantes:
        return 0.0
    top_k = recuperados[:k]
    aciertos = sum(1 for doc_id in top_k if doc_id in relevantes)
    return aciertos / len(relevantes)


def average_precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """Average Precision (AP@k) para una consulta individual."""
    if not relevantes or k <= 0:
        return 0.0

    top_k = recuperados[:k]
    suma_precisiones = 0.0
    aciertos = 0

    for idx, doc_id in enumerate(top_k, start=1):
        if doc_id in relevantes:
            aciertos += 1
            suma_precisiones += aciertos / idx

    return suma_precisiones / len(relevantes)


def _extraer_relevantes(entrada_qrel: Any) -> set[str]:
    """Soporta qrels en formato dict {doc_id: score} o lista [doc_id, ...]."""
    if isinstance(entrada_qrel, dict):
        return {str(doc_id) for doc_id, score in entrada_qrel.items() if score > 0}
    return {str(doc_id) for doc_id in entrada_qrel}


def evaluar_modelo(
    modelo: Any,
    consultas: dict[str, str],
    qrels: dict[str, Any],
    k: int = 10,
) -> dict[str, Any]:
    """
    Evalúa un modelo sobre todo el conjunto de consultas con qrels.
    Retorna métricas globales (MAP, Precision@k, Recall@k, latencia) y detalle por consulta.
    """
    detalle_consultas: dict[str, dict[str, Any]] = {}
    suma_p = 0.0
    suma_r = 0.0
    suma_ap = 0.0
    tiempo_total = 0.0
    n_evaluadas = 0

    for qid, rel_raw in qrels.items():
        if qid not in consultas:
            continue

        relevantes = _extraer_relevantes(rel_raw)
        if not relevantes:
            continue

        texto_q = consultas[qid]

        t0 = time.perf_counter()
        resultados = modelo.buscar(texto_q, k)
        dt = time.perf_counter() - t0
        tiempo_total += dt

        docs_recuperados = [str(doc_id) for doc_id, _ in resultados]

        p = precision_at_k(docs_recuperados, relevantes, k)
        r = recall_at_k(docs_recuperados, relevantes, k)
        ap = average_precision_at_k(docs_recuperados, relevantes, k)

        suma_p += p
        suma_r += r
        suma_ap += ap
        n_evaluadas += 1

        detalle_consultas[qid] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "ap": round(ap, 4),
            "top_docs": docs_recuperados[:5],
        }

    divisor = max(n_evaluadas, 1)
    return {
        "modelo": modelo.nombre,
        "k": k,
        "consultas_evaluadas": n_evaluadas,
        "MAP": round(suma_ap / divisor, 4),
        "Precision_promedio": round(suma_p / divisor, 4),
        "Recall_promedio": round(suma_r / divisor, 4),
        "latencia_promedio_ms": round((tiempo_total / divisor) * 1000, 2),
        "por_consulta": detalle_consultas,
    }


def _snippet_doc(doc_store: dict[str, Any], doc_id: str, max_chars: int = 220) -> str:
    doc = doc_store.get(doc_id, "")
    if isinstance(doc, dict):
        titulo = doc.get("titulo") or doc.get("title") or ""
        texto = doc.get("texto") or doc.get("text") or ""
        contenido = f"{titulo} - {texto}".strip(" -")
    else:
        contenido = str(doc)
    return contenido[:max_chars].replace("\n", " ") + ("..." if len(contenido) > max_chars else "")


def analizar_divergencias(
    evaluaciones: dict[str, dict[str, Any]],
    consultas: dict[str, str],
    qrels: dict[str, Any],
    doc_store: dict[str, Any],
    top_casos: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Identifica casos concretos para la sección f:
    - Donde recuperación semántica supera claramente a BM25 y TF-IDF.
    - Donde recuperación semántica empeora frente a BM25 / TF-IDF.
    """
    nombre_sem = next((n for n in evaluaciones if "embedding" in n.lower() or "sem" in n.lower()), None)
    nombre_bm25 = next((n for n in evaluaciones if "bm25" in n.lower()), None)
    nombre_tfidf = next((n for n in evaluaciones if "tfidf" in n.lower() or "tf-idf" in n.lower()), None)

    if not (nombre_sem and nombre_bm25 and nombre_tfidf):
        return {"mejora_semantica": [], "empeora_semantica": []}

    det_sem = evaluaciones[nombre_sem]["por_consulta"]
    det_bm25 = evaluaciones[nombre_bm25]["por_consulta"]
    det_tfidf = evaluaciones[nombre_tfidf]["por_consulta"]

    diferencias = []
    for qid in det_sem:
        if qid not in det_bm25 or qid not in det_tfidf:
            continue
        ap_sem = det_sem[qid]["ap"]
        ap_lex = max(det_bm25[qid]["ap"], det_tfidf[qid]["ap"])
        delta = ap_sem - ap_lex
        diferencias.append((delta, qid))

    diferencias.sort(key=lambda x: x[0], reverse=True)

    def construir_registro(delta: float, qid: str) -> dict[str, Any]:
        relevantes = sorted(_extraer_relevantes(qrels[qid]))
        return {
            "qid": qid,
            "consulta": consultas[qid],
            "delta_ap_sem_vs_lex": round(delta, 4),
            "relevantes": [
                {"doc_id": did, "extracto": _snippet_doc(doc_store, did)}
                for did in relevantes[:3]
            ],
            "resultados": {
                nombre: {
                    "ap": evaluaciones[nombre]["por_consulta"][qid]["ap"],
                    "recall": evaluaciones[nombre]["por_consulta"][qid]["recall"],
                    "top_3": evaluaciones[nombre]["por_consulta"][qid]["top_docs"][:3],
                }
                for nombre in evaluaciones
            },
        }

    mejoras = [construir_registro(d, qid) for d, qid in diferencias[:top_casos] if d > 0]
    empeora = [construir_registro(d, qid) for d, qid in reversed(diferencias[-top_casos:]) if d < 0]

    return {
        "mejora_semantica": mejoras,
        "empeora_semantica": empeora,
    }