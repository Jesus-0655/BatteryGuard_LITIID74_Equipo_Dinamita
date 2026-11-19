"""Inferencia para baterias NUEVAS con datos parciales (lo que llega desde el Dashboard).

Adaptado del apendice del notebook: en lugar de recalcular referencias desde el CSV,
las carga de models/referencias.joblib (generado por train.py).
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.models import predict as P

ESENCIALES = ["thermal_runaway_risk", "previous_faults", "BMS_warning_count", "abnormal_voltage_events"]
GRUPO_SALUD = ["battery_health_percent", "capacity_loss_percent", "state_of_health"]  # basta UNA
RECOMENDADAS = ["battery_stress_index", "cell_temperature_max", "discharge_efficiency",
                "internal_resistance", "cycle_count", "pack_voltage", "vehicle_age_years"]
INDICES = ["thermal_runaway_risk", "battery_stress_index"]


@lru_cache(maxsize=1)
def _cargar():
    art = P.cargar_artefactos()
    refs = joblib.load(P.MODELS_DIR / "referencias.joblib")
    reg = joblib.load(P.MODELS_DIR / "modelo_ciclos.joblib")
    meta = art["meta"]
    return {"art": art, "refs": refs, "reg": reg, "COLS": meta["columnas_entrada"],
            "NUM": meta["columnas_numericas"], "CAT": meta["columnas_categoricas"],
            "RAW": [c for c in meta["columnas_numericas"] if c not in INDICES]}


def recargar():
    """Se llama tras reentrenar: descarta lo cargado en memoria."""
    _cargar.cache_clear()
    P.cargar_artefactos.cache_clear()


def modelo_disponible() -> bool:
    return all((P.MODELS_DIR / f).exists() for f in
               ["modelo_baterias.joblib", "preprocessor.joblib", "metadata.json",
                "referencias.joblib", "modelo_ciclos.joblib"])


def _es_nulo(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def completar(parcial: pd.DataFrame, min_completitud_indices: float = 0.80):
    c = _cargar(); REF, COLS, RAW = c["refs"]["REF"], c["COLS"], c["RAW"]
    Z = parcial.copy(); derivados, estimados = {}, {}
    if "battery_health_percent" not in Z and "capacity_loss_percent" in Z:
        Z["battery_health_percent"] = 100 - Z["capacity_loss_percent"]
        derivados["battery_health_percent"] = "100 - capacity_loss_percent (exacta)"
    if "battery_health_percent" not in Z and "state_of_health" in Z:
        Z["battery_health_percent"] = Z["state_of_health"]
        derivados["battery_health_percent"] = "≈ state_of_health (correlación 0.96)"
    if "capacity_loss_percent" not in Z and "battery_health_percent" in Z:
        Z["capacity_loss_percent"] = 100 - Z["battery_health_percent"]
        derivados["capacity_loss_percent"] = "100 - battery_health_percent (exacta)"
    if "state_of_health" not in Z and "battery_health_percent" in Z:
        Z["state_of_health"] = Z["battery_health_percent"]
        derivados["state_of_health"] = "≈ battery_health_percent (correlación 0.96)"

    completitud_raw = sum(k in Z for k in RAW) / len(RAW)
    faltan_idx = [t for t in INDICES if t not in Z and t in c["refs"]["estimadores"]]
    if faltan_idx and completitud_raw >= min_completitud_indices:
        base = pd.DataFrame({k: (Z[k] if k in Z else REF[k]) for k in RAW}, index=Z.index)
        for t in faltan_idx:
            Z[t] = c["refs"]["estimadores"][t].predict(base)
            estimados[t] = f"estimado desde las demás variables ({completitud_raw:.0%} proporcionadas)"
    imputados = [k for k in COLS if k not in Z]
    for k in imputados:
        Z[k] = REF[k]
    return Z[COLS], {"derivados": derivados, "estimados": estimados, "imputados": imputados}


def _nivel(n_rec: int) -> str:
    return "Alta" if n_rec >= 6 else "Media" if n_rec >= 3 else "Baja (orientativa)"


def predecir_nueva(muestra, top_n: int = 5) -> dict:
    """Clasifica una bateria (Optimo / Mantenimiento / Descarte) y estima ciclos restantes.
    Lanza ValueError si faltan datos esenciales o hay valores no numericos."""
    c = _cargar(); COLS, NUM, CAT = c["COLS"], c["NUM"], c["CAT"]
    RANGO, CATEGORIAS = c["refs"]["RANGO"], c["refs"]["CATEGORIAS"]
    if isinstance(muestra, str):
        muestra = json.loads(muestra)
    if not isinstance(muestra, dict):
        raise TypeError("La muestra debe ser un dict o un string JSON.")

    dado = {k: muestra[k] for k in COLS if k in muestra and not _es_nulo(muestra[k])}
    adv = []
    for k in NUM:
        if k in dado:
            try:
                dado[k] = float(dado[k])
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' debe ser numérico; se recibió {dado[k]!r}.")
            lo, hi = RANGO[k]
            if not lo <= dado[k] <= hi:
                adv.append(f"{k}={dado[k]} está fuera del rango de entrenamiento [{lo:.4g}, {hi:.4g}]: "
                           f"el modelo extrapola y la predicción es menos fiable.")
    for k in CAT:
        if k in dado and dado[k] not in CATEGORIAS[k]:
            adv.append(f"{k}='{dado[k]}' no existía en el entrenamiento; se trata como desconocida.")

    Z, info = completar(pd.DataFrame([dado]))
    faltan = [k for k in ESENCIALES if k not in dado and k not in info["estimados"]]
    if not any(k in dado for k in GRUPO_SALUD):
        faltan.append("battery_health_percent (o capacity_loss_percent, o state_of_health)")
    if faltan:
        raise ValueError(f"Faltan datos esenciales para responder: {faltan}")

    r = P.predecir(Z.iloc[0].to_dict(), top_n=top_n)
    for e in r["explicacion"]:
        v = e["variable"]
        e["origen"] = ("proporcionado" if v in dado else "derivado" if v in info["derivados"]
                       else "estimado" if v in info["estimados"] else "imputado (media/moda)")
    rec = [k for k in RECOMENDADAS if k in dado or k in info["estimados"]]
    nivel = _nivel(len(rec))
    lo, hi = c["art"]["meta"].get("rango_ciclos") or json.loads(
        (P.MODELS_DIR / "metricas_ciclos.json").read_text())["rango_ciclos"]
    ciclos = float(np.clip(c["reg"].predict(c["art"]["pre"].transform(Z))[0], 0, None))
    r.update({
        "fiabilidad": nivel, "completitud": round(len(dado) / len(COLS), 3),
        "campos_proporcionados": len(dado), "campos_estimados": info["estimados"],
        "campos_imputados": len(info["imputados"]),
        "recomendados_faltantes": [k for k in RECOMENDADAS if k not in rec],
        "advertencias": adv, "ciclos_restantes": round(ciclos), "ciclos_rango": [lo, hi],
    })
    return r
