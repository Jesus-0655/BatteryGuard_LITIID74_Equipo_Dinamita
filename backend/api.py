"""API de BatteryGuard_AI.   Ejecutar:  python -m uvicorn api:app --port 8000
Documentacion interactiva (incluye subir un CSV para reentrenar): http://localhost:8000/docs
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile

from src.models import inference as inf
from src.models import predict as P

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed" / "baterias_procesado.csv"
REQUERIDAS = {"battery_failure", "predicted_remaining_life_cycles"}
ESTADO: dict[str, Any] = {"estado": "inactivo"}

app = FastAPI(title="BatteryGuard_AI")


def _py(o):  # numpy -> tipos nativos de Python
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    return o.item() if hasattr(o, "item") else o


@app.get("/api/salud")
def salud():
    met = {}
    if inf.modelo_disponible():
        met = json.loads((P.MODELS_DIR / "metricas.json").read_text(encoding="utf-8"))
    return {"modelo_cargado": inf.modelo_disponible(), "entrenamiento": ESTADO, "metricas": met}


@app.get("/api/ayudas")
def ayudas():
    """Rangos de referencia por variable y clase (los muestra el Dashboard bajo cada campo)."""
    f = P.MODELS_DIR / "ayudas.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@app.post("/api/predict")
def predict(muestra: dict = Body(...)):
    if not inf.modelo_disponible():
        raise HTTPException(503, "Modelo no entrenado todavía. Sube el dataset en /docs (POST /api/train).")
    try:
        return _py(inf.predecir_nueva(muestra))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))


def _entrenar():
    ESTADO.update(estado="entrenando", inicio=time.time())
    p = subprocess.run([sys.executable, "src/models/train.py", "--data", str(DATA), "--output", "models/"],
                       cwd=BASE, capture_output=True, text=True)
    if p.returncode == 0:
        inf.recargar()
        ESTADO.update(estado="listo", fin=time.time(), log=p.stdout[-1500:])
    else:
        ESTADO.update(estado="error", detalle=(p.stderr or p.stdout)[-1500:])


@app.post("/api/train")
async def train(archivo: UploadFile = File(...)):
    """Sube un CSV (nulos ya imputados) y reentrena ambos modelos en segundo plano."""
    if ESTADO["estado"] == "entrenando":
        raise HTTPException(409, "Ya hay un entrenamiento en curso.")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA.with_suffix(".nuevo.csv")
    tmp.write_bytes(await archivo.read())
    try:
        cols = set(pd.read_csv(tmp, nrows=5).columns)
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el CSV: {e}")
    if not REQUERIDAS <= cols:
        raise HTTPException(400, f"Faltan columnas obligatorias: {sorted(REQUERIDAS - cols)}")
    tmp.replace(DATA)
    threading.Thread(target=_entrenar, daemon=True).start()
    return {"estado": "entrenando", "mensaje": "Consulta el avance en GET /api/salud"}
