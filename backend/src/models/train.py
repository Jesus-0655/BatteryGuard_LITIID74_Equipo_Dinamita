"""Entrenamiento del modelo XGBoost (la comparativa con Regresion Logistica quedo en el notebook de Colab).

Uso:
    python src/models/train.py --data data/processed/baterias_procesado.csv --output models/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, recall_score)
from sklearn.model_selection import (ParameterSampler,
                                     StratifiedKFold, train_test_split)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data import preprocess as pp

LABELS = [0, 1, 2]
NOMBRES = [pp.CLASES[i] for i in LABELS]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def pesos(y, potencia: float):
    """Pesos por muestra: (balanced) ** potencia. 0 = sin pesos, 1 = balanced."""
    return compute_sample_weight("balanced", y) ** potencia


def diagnostico(f1_train: float, f1_test: float) -> str:
    brecha = f1_train - f1_test
    if f1_train < 0.60:
        return "Underfitting (bajo rendimiento incluso en Train)"
    if brecha > 0.10:
        return f"Overfitting (brecha F1 = {brecha:.3f})"
    if brecha > 0.05:
        return f"Sobreajuste moderado (brecha F1 = {brecha:.3f})"
    return f"Ajuste adecuado (brecha F1 = {brecha:.3f})"


def evaluar(nombre, modelo, Xtr, ytr, Xte, yte, t_fit, hiperparametros):
    t0 = time.perf_counter()
    p_te = modelo.predict(Xte)
    t_pred = time.perf_counter() - t0
    p_tr = modelo.predict(Xtr)
    rec = recall_score(yte, p_te, labels=LABELS, average=None, zero_division=0)
    m = {
        "modelo": nombre,
        "hiperparametros": hiperparametros,
        "accuracy_train": accuracy_score(ytr, p_tr),
        "accuracy_test": accuracy_score(yte, p_te),
        "f1_macro_train": f1_score(ytr, p_tr, average="macro"),
        "f1_macro_test": f1_score(yte, p_te, average="macro"),
        "recall_optimo": rec[0],
        "recall_mantenimiento": rec[1],
        "recall_descarte": rec[2],
        "tiempo_entrenamiento_s": t_fit,
        "tiempo_prediccion_test_s": t_pred,
    }
    m["diagnostico"] = diagnostico(m["f1_macro_train"], m["f1_macro_test"])
    return m, p_te


def main(data_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Carga ---------------------------------------------------------------
    log("=== 1. CARGA DE DATOS ===")
    df = pd.read_csv(data_path)
    log(f"Registros cargados: {len(df):,} | Columnas: {df.shape[1]}")
    nulos = pp.verificar_nulos(df)
    log(f"Imputacion de nulos con la media: ya aplicada previamente "
        f"(verificacion: {nulos} NaN restantes)")

    # 2. Etiquetas y variables -----------------------------------------------
    log("=== 2. PREPARACION ===")
    X, y = pp.preparar_datos(df)
    log(f"Umbral de mantenimiento: vida restante < {pp.UMBRAL_MANTENIMIENTO} ciclos")
    log(f"Variables excluidas: {len(pp.columnas_a_excluir(df))} "
        f"(identificacion, *_score, vida restante, falla)")
    log(f"Variables de entrada: {X.shape[1]}")
    dist = y.map(pp.CLASES).value_counts()
    for k, v in dist.items():
        log(f"  Clase {k}: {v:,} ({v / len(y):.1%})")

    # 3. Split ---------------------------------------------------------------
    log("=== 3. DIVISION TRAIN / TEST (80/20 estratificada, seed=42) ===")
    X_tr, X_te, y_tr, y_te = pp.dividir(X, y)
    log(f"Train: {len(X_tr):,} registros | Test: {len(X_te):,} registros")

    pre = pp.construir_preprocesador(X_tr)
    Xtr = pre.fit_transform(X_tr)  # ajustado SOLO con Train (sin fuga)
    Xte = pre.transform(X_te)
    feature_names = list(pre.get_feature_names_out())
    log(f"Escalado + One-Hot ajustados con Train: {Xtr.shape[1]} features")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=pp.SEED)
    resultados = []

    # 4. Modelo principal ----------------------------------------------------
    log("=== 4. PRINCIPAL: XGBoost (busqueda aleatoria con CV estratificada) ===")
    t0 = time.perf_counter()
    espacio = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 300, 500],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [1, 5, 10],
        "gamma": [0, 0.5, 1],
        # 0 = sin ponderar, 0.5 = ponderacion suavizada (raiz), 1 = 'balanced' completo
        "peso_potencia": [0.0, 0.5, 1.0],
    }
    N_ITER = 15
    log(f"{N_ITER} combinaciones x 3 folds = {N_ITER * 3} entrenamientos (metrica: F1 macro sin ponderar)")
    y_np = y_tr.to_numpy()
    mejor_score, mejor_p = -1.0, None
    for i, p in enumerate(ParameterSampler(espacio, n_iter=N_ITER, random_state=pp.SEED), 1):
        hp = {k: v for k, v in p.items() if k != "peso_potencia"}
        scores = []
        for idx_tr, idx_va in cv.split(Xtr, y_np):
            w = pesos(y_np[idx_tr], p["peso_potencia"])
            m = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                              random_state=pp.SEED, n_jobs=-1, **hp)
            m.fit(Xtr[idx_tr], y_np[idx_tr], sample_weight=w)
            scores.append(f1_score(y_np[idx_va], m.predict(Xtr[idx_va]), average="macro"))
        sc = float(np.mean(scores))
        log(f"  [{i:>2}/{N_ITER}] F1 macro CV = {sc:.4f} | depth={p['max_depth']} lr={p['learning_rate']} "
            f"n={p['n_estimators']} peso^{p['peso_potencia']}")
        if sc > mejor_score:
            mejor_score, mejor_p = sc, p
    log(f"Mejores hiperparametros: {mejor_p}")
    log(f"F1 macro CV (3 folds) = {mejor_score:.4f}")
    potencia = mejor_p["peso_potencia"]
    mejor_hp = {k: v for k, v in mejor_p.items() if k != "peso_potencia"}

    log("Refit final con Early Stopping (10% de Train como validacion, paciencia=30)")
    X_a, X_v, y_a, y_v = train_test_split(
        Xtr, y_tr, test_size=0.10, stratify=y_tr, random_state=pp.SEED)
    sw_a = pesos(y_a.to_numpy(), potencia)
    sw_v = pesos(y_v.to_numpy(), potencia)
    params = {**mejor_hp, "n_estimators": 1000}
    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", random_state=pp.SEED, n_jobs=-1,
        early_stopping_rounds=30, **params,
    )
    xgb.fit(X_a, y_a, sample_weight=sw_a, eval_set=[(X_v, y_v)],
            sample_weight_eval_set=[sw_v], verbose=False)
    t_xgb = time.perf_counter() - t0
    best_it = int(xgb.best_iteration) + 1
    log(f"Early stopping: se detuvo en {best_it} arboles (de 1000 posibles)")
    hp_final = {**mejor_p, "n_estimators_efectivos": best_it, "early_stopping_rounds": 30}
    m_xgb, p_xgb = evaluar("XGBoost (principal)", xgb, Xtr, y_tr, Xte, y_te, t_xgb, hp_final)
    resultados.append(m_xgb)
    log(f"XGBoost -> Test Acc {m_xgb['accuracy_test']:.4f} | "
        f"F1 macro {m_xgb['f1_macro_test']:.4f}")

    # 5. Reporte final -------------------------------------------------------
    log("=== 5. REPORTE FINAL EN TEST SET ===")
    print("\n--- XGBoost ---")
    print(classification_report(y_te, p_xgb, labels=LABELS, target_names=NOMBRES, digits=4))
    for m in resultados:
        log(f"{m['modelo']}: Train F1 {m['f1_macro_train']:.4f} vs Test F1 "
            f"{m['f1_macro_test']:.4f} -> {m['diagnostico']}")

    # 6. Guardado de artefactos ---------------------------------------------
    log("=== 6. GUARDADO DE ARTEFACTOS ===")
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    metadata = {
        "clases": {str(k): v for k, v in pp.CLASES.items()},
        "umbral_mantenimiento": pp.UMBRAL_MANTENIMIENTO,
        "columnas_entrada": list(X.columns),
        "columnas_numericas": num_cols,
        "columnas_categoricas": cat_cols,
        "categorias": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
        "features_transformadas": feature_names,
        "medias_entrenamiento": {c: float(X_tr[c].mean()) for c in num_cols},
    }
    joblib.dump(xgb, out / "modelo_baterias.joblib")
    joblib.dump(pre, out / "preprocessor.joblib")
    from src.models.extras import guardar_extras  # regresor de ciclos + referencias
    met_c = guardar_extras(df, X, X_tr, X_te, Xtr, Xte, out)
    log(f"Regresor de ciclos restantes: MAE = {met_c['mae_ciclos']:.1f} ciclos | R2 = {met_c['r2']:.3f}")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metricas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    test_dir = Path(data_path).resolve().parent
    test_df = X_te.copy()
    test_df["estado_bateria"] = y_te
    test_df.to_csv(test_dir / "test_set.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, p_xgb, labels=LABELS, display_labels=NOMBRES, ax=ax,
        cmap="Blues", colorbar=False, xticks_rotation=20)
    ax.set_title("XGBoost")
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion.png", dpi=130)
    log(f"Artefactos guardados en: {out}/")
    log("Entrenamiento finalizado correctamente.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="CSV procesado (nulos ya imputados)")
    ap.add_argument("--output", default="models/", help="Carpeta de artefactos")
    args = ap.parse_args()
    main(args.data, args.output)
