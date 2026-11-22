import html
import time
from collections import Counter
from pathlib import Path
from typing import Any

import config
from src import almacen_vectores
from src.data_loader import cargar_json
from src.indice_invertido import IndiceInvertido
from src.modelos.bm25 import ModeloBM25
from src.modelos.modelo_binario import ModeloJaccard
from src.modelos.semantico import GeneradorEmbeddings, ModeloSemantico
from src.modelos.tfidf_coseno import ModeloTfidfCoseno


class ServicioBusqueda:
    def __init__(self) -> None:
        self.indice = IndiceInvertido.cargar(config.RUTA_INDICE_INVERTIDO)
        self.faiss_idx, self.doc_ids = almacen_vectores.cargar(
            config.RUTA_FAISS, config.RUTA_DOC_IDS
        )
        self.doc_store: dict[str, Any] = cargar_json(config.RUTA_DOC_STORE)
        self.consultas: dict[str, str] = cargar_json(config.RUTA_CONSULTAS)

        self.generador = GeneradorEmbeddings()
        self.modelos = {
            "binario": ModeloJaccard(self.indice),
            "tfidf": ModeloTfidfCoseno(self.indice),
            "bm25": ModeloBM25(self.indice),
            "embeddings": ModeloSemantico(self.generador, self.faiss_idx, self.doc_ids),
        }

        self.metricas = self._cargar_opcional(
            [config.RUTA_METRICAS, config.DIR_ARTEFACTOS / "metricas.json"], default={}
        )
        self.comparativo = self._cargar_opcional(
            [config.DIR_ARTEFACTOS / "analisis_comparativo.json"], default={}
        )

    @staticmethod
    def _cargar_opcional(rutas: list[Path], default: Any) -> Any:
        for ruta in rutas:
            if Path(ruta).exists():
                return cargar_json(ruta)
        return default

    def _formatear_doc(self, doc_id: str, score: float, max_chars: int = 360) -> dict[str, Any]:
        raw = self.doc_store.get(str(doc_id), {})
        if isinstance(raw, dict):
            titulo = html.unescape(str(raw.get("titulo") or raw.get("title") or "")).strip()
            texto = html.unescape(str(raw.get("texto") or raw.get("text") or "")).strip()
        else:
            titulo = ""
            texto = html.unescape(str(raw)).strip()

        extracto = texto[:max_chars] + ("..." if len(texto) > max_chars else "")
        return {
            "doc_id": str(doc_id),
            "score": round(float(score), 4),
            "titulo": titulo,
            "extracto": extracto,
        }

    def info(self) -> dict[str, Any]:
        ejemplos = [
            "Where should I park my rainy-day / emergency fund?",
            "What approaches are there for pricing a small business?",
            "Employer rollover from 403b to 401k?",
            "How does pre-market trading work?",
            "Is there a way to open a U.S. bank account for my LLC remotely?",
            "capital gains tax on stock dividends",
        ]
        return {
            "corpus_size": len(self.doc_store),
            "total_consultas_eval": len(self.consultas),
            "modelos": [
                {"clave": k, "nombre": m.nombre} for k, m in self.modelos.items()
            ],
            "ejemplos": ejemplos,
        }

    def buscar(self, consulta: str, clave_modelo: str = "bm25", top_n: int = 10) -> dict[str, Any]:
        if clave_modelo not in self.modelos:
            raise ValueError(f"Modelo inválido: '{clave_modelo}'. Opciones: {list(self.modelos)}")

        consulta = (consulta or "").strip()
        top_n = max(1, min(int(top_n), 50))
        modelo = self.modelos[clave_modelo]

        if not consulta:
            return {
                "consulta": "",
                "modelo": modelo.nombre,
                "clave_modelo": clave_modelo,
                "latencia_ms": 0.0,
                "resultados": [],
            }

        t0 = time.perf_counter()
        raw_res = modelo.buscar(consulta, top_n)
        latencia_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "consulta": consulta,
            "modelo": modelo.nombre,
            "clave_modelo": clave_modelo,
            "latencia_ms": latencia_ms,
            "resultados": [self._formatear_doc(d, s) for d, s in raw_res],
        }

    def comparar(self, consulta: str, top_n: int = 5) -> dict[str, Any]:
        consulta = (consulta or "").strip()
        top_n = max(1, min(int(top_n), 20))

        resultados_por_modelo = {}
        conteo_docs: Counter[str] = Counter()

        for clave in self.modelos:
            res = self.buscar(consulta, clave, top_n)
            resultados_por_modelo[clave] = res
            for item in res["resultados"]:
                conteo_docs[item["doc_id"]] += 1

        compartidos = {did: n for did, n in conteo_docs.items() if n > 1}
        return {
            "consulta": consulta,
            "top_n": top_n,
            "compartidos": compartidos,
            "modelos": resultados_por_modelo,
        }

    def obtener_metricas(self) -> dict[str, Any]:
        resumen = []
        for nombre, datos in self.metricas.items():
            resumen.append(
                {
                    "modelo": datos.get("modelo", nombre),
                    "k": datos.get("k", config.TOP_K),
                    "consultas_evaluadas": datos.get("consultas_evaluadas", 0),
                    "MAP": datos.get("MAP", 0.0),
                    "Precision_promedio": datos.get("Precision_promedio", 0.0),
                    "Recall_promedio": datos.get("Recall_promedio", 0.0),
                    "latencia_promedio_ms": datos.get("latencia_promedio_ms", 0.0),
                }
            )
        return {
            "resumen": resumen,
            "comparativo": self.comparativo,
        }