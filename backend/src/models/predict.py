"""Inferencia desacoplada con explicabilidad (SHAP).

Los artefactos (modelo, preprocesador, metadatos y explainer SHAP) se cargan en
memoria UNA sola vez (lru_cache); las llamadas posteriores a `predecir()` los
reutilizan.

Ejemplo:
    from src.models.predict import predecir
    resultado = predecir({...}) # dict o string JSON con las variables de la bateria
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@lru_cache(maxsize=1)
def cargar_artefactos(models_dir: str = str(MODELS_DIR)) -> dict:
    d = Path(models_dir)
    modelo = joblib.load(d / "modelo_baterias.joblib")
    return {
        "modelo": modelo,
        "pre": joblib.load(d / "preprocessor.joblib"),
        "meta": json.loads((d / "metadata.json").read_text(encoding="utf-8")),
        "explainer": shap.TreeExplainer(modelo),
    }


def _shap_de_clase(shap_values, clase: int) -> np.ndarray:
    """Normaliza la salida de SHAP (cambia segun la version de la libreria)."""
    if isinstance(shap_values, list):  # shap < 0.45: lista [clase] -> (n, f)
        return np.asarray(shap_values[clase])[0]
    arr = np.asarray(shap_values)  # shap >= 0.45: (n, f, clases)
    return arr[0, :, clase] if arr.ndim == 3 else arr[0]


def _valor_base(explainer, clase: int) -> float:
    ev = np.atleast_1d(explainer.expected_value)
    return float(ev[clase] if len(ev) > 1 else ev[0])


def predecir(muestra, top_n: int = 5, models_dir: str | None = None) -> dict:
    """Predice el estado de una bateria y explica el porque.

    Parametros
    ----------
    muestra : dict | str(JSON) con las variables de entrada de UNA bateria.
              Se ignoran claves extra (id, marca, etc.).
    top_n   : numero de variables mas influyentes a devolver.
    """
    art = cargar_artefactos(models_dir) if models_dir else cargar_artefactos()
    meta, pre, modelo, explainer = art["meta"], art["pre"], art["modelo"], art["explainer"]

    if isinstance(muestra, str):
        muestra = json.loads(muestra)
    if not isinstance(muestra, dict):
        raise TypeError("La muestra debe ser un dict o un string JSON.")

    cols = meta["columnas_entrada"]
    faltantes = [c for c in cols if c not in muestra]
    if faltantes:
        raise ValueError(f"Faltan {len(faltantes)} variables requeridas: {faltantes}")
    nulos = [c for c in cols if muestra[c] is None or (isinstance(muestra[c], float) and np.isnan(muestra[c]))]
    if nulos:
        raise ValueError(f"Valores nulos no permitidos en: {nulos}")

    fila = pd.DataFrame([{c: muestra[c] for c in cols}])
    Xt = pre.transform(fila)
    proba = modelo.predict_proba(Xt)[0]
    clase = int(np.argmax(proba))
    etiqueta = meta["clases"][str(clase)]

    # --- Explicabilidad: SHAP de la clase predicha, agregado por variable original
    sv = _shap_de_clase(explainer.shap_values(Xt), clase)
    cat_cols = meta["columnas_categoricas"]
    agregado = {}
    for nombre, valor in zip(meta["features_transformadas"], sv):
        origen = next((c for c in cat_cols if nombre.startswith(c + "_")), nombre)
        agregado[origen] = agregado.get(origen, 0.0) + float(valor)

    total_abs = sum(abs(v) for v in agregado.values()) or 1.0
    top = sorted(agregado.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    explicacion = []
    for var, contrib in top:
        item = {
            "variable": var,
            "valor": muestra[var],
            "contribucion_shap": round(contrib, 4),
            "impacto_relativo_pct": round(100 * abs(contrib) / total_abs, 1),
            "efecto": f"{'empuja hacia' if contrib > 0 else 'aleja de'} '{etiqueta}'",
        }
        if var in meta["medias_entrenamiento"]:
            item["promedio_flota"] = round(meta["medias_entrenamiento"][var], 3)
        explicacion.append(item)

    factores = "; ".join(
        f"{e['variable']}={e['valor']} ({e['efecto'].split(' ')[0]}, {e['impacto_relativo_pct']}%)"
        for e in explicacion[:3]
    )
    return {
        "clase_predicha": clase,
        "etiqueta": etiqueta,
        "confianza": round(float(proba[clase]), 4),
        "probabilidades": {meta["clases"][str(i)]: round(float(p), 4) for i, p in enumerate(proba)},
        "explicacion": explicacion,
        "valor_base_shap": round(_valor_base(explainer, clase), 4),
        "resumen": f"Bateria clasificada como '{etiqueta}' con {proba[clase]:.1%} de confianza. "
                   f"Factores principales: {factores}.",
    }
