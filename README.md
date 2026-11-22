# Sistema de Recuperación de Información

Proyecto de 1er Bimestre — **Recuperación de Información**  
**Docente:** Prof. Iván Carrera  

Sistema de recuperación de información que permite indexar un corpus textual financiero, ejecutar consultas de texto libre y comparar cuantitativa y cualitativamente cuatro modelos de recuperación (tres clásicos basados en índice invertido propio y uno semántico basado en embeddings y FAISS).

> **URL del despliegue en producción:** `https://proyecto-ri.onrender.com`  

## Características principales

* **Preprocesamiento e índice invertido propio:** Tokenización alfanumérica, normalización a minúsculas, remoción de *stopwords* en inglés (NLTK) y *stemming* con `PorterStemmer`. El índice almacena las listas de *postings* con frecuencias por término, longitudes de documento y normas TF-IDF.
* **Modelos clásicos implementados desde cero:**
  * **Binario (Jaccard):** similitud de Jaccard sobre conjuntos binarios de términos.
  * **Vectorial (TF-IDF):** similitud de coseno con ponderación TF-IDF.
  * **Probabilístico (BM25):** esquema Okapi BM25 con hiperparámetros $k_1 = 1.5$ y $b = 0.75$.
* **Recuperación semántica con embeddings:** generación de representaciones densas de 384 dimensiones normalizadas ($L_2 = 1.0$) mediante `all-MiniLM-L6-v2` (vía `fastembed` / ONNX) y búsqueda vectorial con **FAISS**.
* **Evaluación estandarizada:** cálculo de *Precision@k* y *Recall@k* por consulta y *MAP@k* global sobre `qrels`, junto con extracción automática de casos de divergencia entre recuperación semántica y léxica.
* **Interfaz web (FastAPI):** permite ejecutar consultas libres eligiendo el modelo y el `top-n`, comparar los 4 modelos simultáneamente resaltando documentos compartidos, desplegar el texto completo de cada documento y visualizar métricas y gráficos comparativos.

---

## Corpus utilizado

Se utiliza un subconjunto reproducible (semilla `42`) del dataset **FiQA** (`mteb/fiqa` de Hugging Face), orientado a preguntas y respuestas sobre finanzas personales:

* **Documentos indexados:** 20,000 documentos (garantizando la inclusión del 100% de los documentos relevantes del conjunto de prueba).
* **Consultas de evaluación:** 648 consultas (split `test`).
* **Juicios de relevancia (`qrels`):** 1,706 pares consulta–documento relevante.
* **Vocabulario del índice invertido:** 31,080 términos únicos (longitud promedio de $71.2$ tokens por documento).

---

## Estructura del proyecto

```text
Proyecto_Recuperacion_de_Informacion/
├── app.py                        # Servidor backend FastAPI y endpoints API
├── config.py                     # Rutas de artefactos e hiperparámetros globales
├── requirements.txt              # Dependencias de producción
├── pruebas.ipynb                 # Notebook de validaciones técnicas y análisis
├── static/
│   └── index.html                # Interfaz web (Búsqueda, Comparación y Métricas)
├── scripts/
│   ├── crear_indice.py           # Descarga FiQA, preprocesa y construye el índice invertido
│   ├── crear_embeddings.py       # Genera embeddings de documentos y construye faiss.index
│   └── evaluacion.py             # Evalúa los 4 modelos (Precision, Recall, MAP) y exporta JSON
├── src/
│   ├── data_loader.py            # Carga del dataset FiQA y utilidades JSON
│   ├── preprocesamiento.py       # Tokenización, stopwords y PorterStemmer
│   ├── indice_invertido.py       # Estructura del índice invertido propio
│   ├── almacen_vectores.py       # Persistencia y carga del índice FAISS
│   ├── evaluacion_modelo.py      # Métricas (P@k, R@k, AP@k, MAP) y análisis de divergencias
│   ├── servicio_busqueda.py      # Capa de negocio que conecta modelos y artefactos con la web
│   └── modelos/
│       ├── base.py               # Interfaz base para los modelos de recuperación
│       ├── modelo_binario.py     # Modelo Binario (Jaccard)
│       ├── tfidf_coseno.py       # Modelo Vectorial (TF-IDF + Coseno)
│       ├── bm25.py               # Modelo Probabilístico (BM25)
│       └── semantico.py          # Generador de embeddings y búsqueda vectorial en FAISS
└── artifacts/                    # Artefactos precalculados (índices, corpus, qrels y métricas)
    ├── inverted_index.pkl
    ├── faiss.index
    ├── doc_ids.json
    ├── doc_store.json
    ├── consultas.json
    ├── qrels.json
    ├── metricas.json
    └── analisis_comparativo.json
```

---

## Resultados de evaluación ($k = 10$, 648 consultas)

| Modelo | MAP@10 | Precision@10 | Recall@10 | Latencia promedio (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Binario (Jaccard)** | 0.0478 | 0.0258 | 0.1064 | 7.63 |
| **Vectorial (TF-IDF)** | 0.1868 | 0.0755 | 0.3328 | 12.70 |
| **Probabilístico (BM25)** | 0.2640 | 0.0926 | 0.4134 | 12.74 |
| **Vectorial (Embeddings)** | **0.3884** | **0.1309** | **0.5455** | 17.73 |

---

## Requisitos e instalación local

* **Python:** 3.10 o superior.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/JesuaVAlc/Proyecto_Recuperacion_de_Informacion.git
   cd Proyecto_Recuperacion_de_Informacion
   ```

2. **Crear y activar un entorno virtual:**
   * En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * En Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Si deseas regenerar el corpus desde Hugging Face ejecutando `scripts.crear_indice`, instala adicionalmente `pip install datasets`).*

---

## Instrucciones de ejecución

### 1. Ejecutar la interfaz web (usando los artefactos ya incluidos)

El repositorio incluye los índices y métricas precalculados en `artifacts/`. Para iniciar la aplicación directamente:

```bash
python -m uvicorn app:app --reload
```

Luego abre en tu navegador: `http://127.0.0.1:8000`

### 2. Regenerar índices, embeddings y evaluación desde cero (opcional)

Si deseas reconstruir todo el pipeline completo:

```bash
# 1. Construir índice invertido y guardar corpus/consultas/qrels
python -m scripts.crear_indice

# 2. Generar embeddings de los 20,000 documentos y construir el índice FAISS
python -m scripts.crear_embeddings

# 3. Ejecutar la evaluación global (Precision@10, Recall@10, MAP@10) y casos comparativos
python -m scripts.evaluacion
```

---

## Despliegue en Render

El proyecto está configurado para desplegarse como un **Web Service** en el plan gratuito de Render:

* **Runtime:** `Python 3`
* **Build Command:**
  ```bash
  pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords', quiet=True); from src.modelos.semantico import GeneradorEmbeddings; GeneradorEmbeddings()"
  ```
* **Start Command:**
  ```bash
  python -m uvicorn app:app --host 0.0.0.0 --port $PORT
  ```