from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.servicio_busqueda import ServicioBusqueda

RAIZ = Path(__file__).resolve().parent
DIR_STATIC = RAIZ / "static"

servicio: ServicioBusqueda | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global servicio
    servicio = ServicioBusqueda()
    yield


app = FastAPI(title="Sistema de Recuperación de Información", lifespan=lifespan)

if DIR_STATIC.exists():
    app.mount("/static", StaticFiles(directory=DIR_STATIC), name="static")


def _get_servicio() -> ServicioBusqueda:
    if servicio is None:
        raise HTTPException(status_code=503, detail="Servicio inicializándose.")
    return servicio


@app.get("/")
def index():
    return FileResponse(DIR_STATIC / "index.html")


@app.get("/api/info")
def api_info():
    return _get_servicio().info()


@app.get("/api/buscar")
def api_buscar(
    q: str = Query(..., min_length=1),
    modelo: str = Query("bm25"),
    top_n: int = Query(10, ge=1, le=50),
):
    try:
        return _get_servicio().buscar(q, modelo, top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/comparar")
def api_comparar(
    q: str = Query(..., min_length=1),
    top_n: int = Query(5, ge=1, le=20),
):
    return _get_servicio().comparar(q, top_n)


@app.get("/api/metricas")
def api_metricas():
    return _get_servicio().obtener_metricas()