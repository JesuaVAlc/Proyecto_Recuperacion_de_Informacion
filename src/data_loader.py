"""Carga de FiQA (formato MTEB) + qrels, y submuestreo del corpus."""
import json
import random
from pathlib import Path



import config


# ---------- Utilidades ----------
def _campo(fila, *nombres):
    """Devuelve el primer campo existente (tolera variaciones de nombre)."""
    for nombre in nombres:
        if nombre in fila and fila[nombre] is not None:
            return fila[nombre]
    raise KeyError(f"Ninguna de las columnas {nombres} existe en {list(fila.keys())}")


def _cargar_split(configuracion, split_preferido):
    from datasets import load_dataset
    ds = load_dataset(config.DATASET_HF, configuracion)
    if split_preferido in ds:
        return ds[split_preferido]
    return ds[next(iter(ds.keys()))]  # si el split tiene otro nombre, toma el único


def guardar_json(obj, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def texto_documento(doc):
    """Texto único del documento (título + cuerpo) usado para indexar."""
    return f"{doc['titulo']} {doc['texto']}".strip()


# ---------- Carga desde Hugging Face ----------
def cargar_corpus_completo():
    """Retorna {doc_id: {'titulo': str, 'texto': str}}."""
    corpus = {}
    for fila in _cargar_split("corpus", "corpus"):
        doc_id = str(_campo(fila, "_id", "id"))
        corpus[doc_id] = {
            "titulo": (fila.get("title") or "").strip(),
            "texto": (fila.get("text") or "").strip(),
        }
    return corpus


def cargar_qrels(split=None):
    from datasets import load_dataset
    """Retorna {query_id: {doc_id: relevancia}} solo con relevancia > 0."""
    split = split or config.SPLIT_QRELS
    ds = load_dataset(config.DATASET_HF, "default")[split]
    qrels = {}
    for fila in ds:
        if int(_campo(fila, "score")) <= 0:
            continue
        qid = str(_campo(fila, "query-id", "query_id"))
        did = str(_campo(fila, "corpus-id", "corpus_id"))
        qrels.setdefault(qid, {})[did] = int(_campo(fila, "score"))
    return qrels


def cargar_consultas(ids_validos):
    """Retorna {query_id: texto} solo para las consultas indicadas."""
    consultas = {}
    for fila in _cargar_split("queries", "queries"):
        qid = str(_campo(fila, "_id", "id"))
        if qid in ids_validos:
            consultas[qid] = fila["text"].strip()
    return consultas


# ---------- Submuestreo ----------
def submuestrear_corpus(corpus, qrels, tamano, semilla=config.SEMILLA):
    """
    Conserva TODOS los documentos relevantes de los qrels y completa con
    documentos aleatorios (distractores) hasta alcanzar 'tamano'.
    """
    if tamano is None or tamano >= len(corpus):
        return dict(corpus)

    relevantes = {d for docs in qrels.values() for d in docs if d in corpus}
    restantes = sorted(set(corpus) - relevantes)
    n_extra = max(0, tamano - len(relevantes))

    rng = random.Random(semilla)
    distractores = set(rng.sample(restantes, min(n_extra, len(restantes))))
    elegidos = sorted(relevantes | distractores)
    return {d: corpus[d] for d in elegidos}


def filtrar_qrels(qrels, corpus):
    """Elimina del qrels los documentos ausentes del corpus y consultas vacías."""
    filtrado = {}
    for qid, docs in qrels.items():
        docs_ok = {d: r for d, r in docs.items() if d in corpus}
        if docs_ok:
            filtrado[qid] = docs_ok
    return filtrado


# ---------- Punto de entrada ----------
def cargar_datos(tamano=config.CORPUS_SIZE):
    """Retorna (corpus, consultas, qrels) ya submuestreados y consistentes."""
    corpus_completo = cargar_corpus_completo()
    qrels = cargar_qrels(config.SPLIT_QRELS)
    corpus = submuestrear_corpus(corpus_completo, qrels, tamano)
    qrels = filtrar_qrels(qrels, corpus)
    consultas = cargar_consultas(set(qrels))
    qrels = {q: d for q, d in qrels.items() if q in consultas}
    return corpus, consultas, qrels