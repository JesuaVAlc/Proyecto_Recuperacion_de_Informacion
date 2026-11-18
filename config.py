from pathlib import Path

# ---------- Rutas ----------
RAIZ = Path(__file__).resolve().parent
DIR_ARTEFACTOS = RAIZ / "artifacts"
DIR_CACHE_MODELO = DIR_ARTEFACTOS / "modelo_cache"

RUTA_INDICE_INVERTIDO = DIR_ARTEFACTOS / "inverted_index.pkl"
RUTA_DOC_STORE = DIR_ARTEFACTOS / "doc_store.json"
RUTA_CONSULTAS = DIR_ARTEFACTOS / "consultas.json"
RUTA_QRELS = DIR_ARTEFACTOS / "qrels.json"
RUTA_FAISS = DIR_ARTEFACTOS / "faiss.index"
RUTA_DOC_IDS = DIR_ARTEFACTOS / "doc_ids.json"
RUTA_METRICAS = DIR_ARTEFACTOS / "metrics.json"

# ---------- Datos ----------
DATASET_HF = "mteb/fiqa"
SPLIT_QRELS = "test"
CORPUS_SIZE = 20_000        # None => corpus completo (~57k)
SEMILLA = 42

# ---------- Modelos clásicos ----------
BM25_K1 = 1.5
BM25_B = 0.75

# ---------- Modelo semántico ----------
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
TAMANO_LOTE_EMBEDDINGS = 64

# ---------- Búsqueda ----------
TOP_K = 10