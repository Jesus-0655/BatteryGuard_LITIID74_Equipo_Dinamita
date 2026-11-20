"""Artefactos adicionales que se generan junto al clasificador:

1. modelo_ciclos.joblib  -> regresor XGBoost de 'predicted_remaining_life_cycles'
2. referencias.joblib    -> medias/modas, rangos, categorias y estimadores de los indices,
                            para completar datos faltantes SIN necesitar el CSV en produccion.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.data import preprocess as pp

INDICES = ["thermal_runaway_risk", "battery_stress_index"]

# Variables que el Dashboard muestra con indicador de referencia.
# DIAGNOSTICO: se muestra el rango central (P25-P75) por clase. CONTEXTO: solo mediana y rango total.
AYUDAS_DIAGNOSTICO = ["battery_health_percent", "BMS_warning_count", "previous_faults", "abnormal_voltage_events",
                      "sensor_fault_count", "thermal_runaway_risk", "cell_temperature_max", "discharge_efficiency",
                      "internal_resistance", "cycle_count", "pack_voltage", "cell_voltage_avg",
                      "voltage_imbalance", "last_service_days"]
# Diferencia entre medias de clase (en desviaciones estandar) a partir de la cual se muestra semaforo por clase
UMBRAL_SEPARACION = 0.3
AYUDAS_CONTEXTO = ["state_of_charge", "battery_capacity_kwh", "odometer_km", "daily_distance",
                   "charging_cycles_last_month"]


def guardar_ayudas(df, X, out):
    """models/ayudas.json: rango central (P25-P75) de cada variable por clase real + min/max global."""
    y = pp.construir_etiquetas(df)
    ayudas = {}
    for c in AYUDAS_DIAGNOSTICO + AYUDAS_CONTEXTO:
        if c not in X.columns:
            continue
        por, separa = {}, None
        if c in AYUDAS_DIAGNOSTICO:
            medias = [float(X.loc[y == k, c].mean()) for k in pp.CLASES]
            sd = float(X[c].std()) or 1.0
            separa = (max(medias) - min(medias)) / sd >= UMBRAL_SEPARACION  # ¿distingue las clases por si sola?
        if separa:
            for k, nombre in pp.CLASES.items():
                s = X.loc[y == k, c]
                if len(s):
                    por[nombre] = [round(float(s.quantile(0.25)), 2), round(float(s.quantile(0.75)), 2)]
        ayudas[c] = {"min": round(float(X[c].min()), 2), "max": round(float(X[c].max()), 2),
                     "mediana": round(float(X[c].median()), 2), "separa": separa, "por_clase": por}
    (out / "ayudas.json").write_text(json.dumps(ayudas, indent=2, ensure_ascii=False), encoding="utf-8")


def guardar_extras(df, X, X_tr, X_te, Xtr, Xte, out):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]

    # 1. Regresor de ciclos restantes (misma particion y mismas features que el clasificador)
    ytr = df.loc[X_tr.index, pp.COL_VIDA].to_numpy()
    yte = df.loc[X_te.index, pp.COL_VIDA].to_numpy()
    reg = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, tree_method="hist", random_state=pp.SEED, n_jobs=-1)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    met = {"mae_ciclos": float(mean_absolute_error(yte, pred)), "r2": float(r2_score(yte, pred)),
           "rango_ciclos": [float(ytr.min()), float(ytr.max())]}
    joblib.dump(reg, out / "modelo_ciclos.joblib")
    (out / "metricas_ciclos.json").write_text(json.dumps(met, indent=2), encoding="utf-8")

    # 2. Referencias para completar datos faltantes
    raw = [c for c in num if c not in INDICES]
    refs = {
        "REF": {c: (X_tr[c].mode()[0] if c in cat else float(X_tr[c].mean())) for c in X.columns},
        "RANGO": {c: (float(X_tr[c].min()), float(X_tr[c].max())) for c in num},
        "CATEGORIAS": {c: sorted(X_tr[c].unique().tolist()) for c in cat},
        "estimadores": {t: HistGradientBoostingRegressor(random_state=pp.SEED).fit(X_tr[raw], X_tr[t])
                        for t in INDICES if t in X_tr},
    }
    joblib.dump(refs, out / "referencias.joblib")
    guardar_ayudas(df, X, out)
    return met
