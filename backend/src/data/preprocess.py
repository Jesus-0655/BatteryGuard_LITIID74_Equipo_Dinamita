"""Preprocesamiento: construccion de etiquetas, seleccion de variables,
particion Train/Test y transformador (escalado + one-hot).

Nota: el dataset ya llega con los nulos imputados con la media (paso cumplido
previamente), por eso aqui solo se verifica que no existan NaN.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Definicion de la variable objetivo (3 clases) ---------------------------
COL_FALLA = "battery_failure"
COL_VIDA = "predicted_remaining_life_cycles"
UMBRAL_MANTENIMIENTO = 6000  # ciclos de vida restante; ajustable
CLASES = {0: "Óptimo", 1: "Mantenimiento preventivo", 2: "Descarte"}

# --- Columnas que NO entran al modelo ---------------------------------------
COLS_IDENTIFICACION = [
    "vehicle_id", "vehicle_brand", "vehicle_model", "vehicle_type",
    "battery_serial", "manufacturing_year",
]
# Ademas se excluyen: todas las columnas *_score, COL_VIDA y COL_FALLA
# (con ellas se construye la etiqueta o son indices derivados -> data leakage).

SEED = 42
TEST_SIZE = 0.20


def verificar_nulos(df: pd.DataFrame) -> int:
    """El dataset ya fue imputado con la media; solo se valida."""
    n = int(df.isna().sum().sum())
    if n:
        raise ValueError(f"El dataset contiene {n} nulos; debe venir imputado.")
    return n


def construir_etiquetas(df: pd.DataFrame) -> pd.Series:
    """0 = Óptimo, 1 = Mantenimiento preventivo, 2 = Descarte.

    Descarte      : battery_failure == 1
    Mantenimiento : sin falla y vida restante < UMBRAL_MANTENIMIENTO
    Óptimo        : el resto
    """
    y = pd.Series(0, index=df.index, name="estado_bateria", dtype="int64")
    y[df[COL_VIDA] < UMBRAL_MANTENIMIENTO] = 1
    y[df[COL_FALLA] == 1] = 2
    return y


def columnas_a_excluir(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c in COLS_IDENTIFICACION or c.endswith("_score") or c in (COL_FALLA, COL_VIDA)
    ]


def preparar_datos(df: pd.DataFrame):
    y = construir_etiquetas(df)
    X = df.drop(columns=columnas_a_excluir(df))
    return X, y


def dividir(X: pd.DataFrame, y: pd.Series):
    """Particion estratificada 80/20 con semilla fija (reproducible)."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    )
