import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.figure

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index
]:  # Añadido pd.Index al retorno
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y devuelve también los nombres de las columnas de X_train.

    Args:
        data: El DataFrame de entrada (model_input_data).
        parameters: Diccionario de parámetros de la pipeline (ej. test_size, random_state).

    Returns:
        Una tupla de DataFrames y Series: (X_train, X_test, y_train, y_test, X_train_columns).
    """
    log.info("--- Iniciando división de datos en entrenamiento y prueba ---")

    X = data.drop(parameters["target_column"], axis=1)
    y = data[parameters["target_column"]]

    log.info(f"Dimensiones de X (características): {X.shape}")
    log.info(f"Dimensiones de y (variable objetivo): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parameters["test_size"],
        random_state=parameters["random_state"],
        stratify=y,
    )

    X_train_columns = X_train.columns  # Obtenemos los nombres de las columnas aquí

    log.info(f"Dimensiones de X_train: {X_train.shape}")
    log.info(f"Dimensiones de X_test: {X_test.shape}")
    log.info(f"Dimensiones de y_train: {y_train.shape}")
    log.info(f"Dimensiones de y_test: {y_test.shape}")
    log.info("--- División de datos completada ---")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_columns,
    )  # Devolvemos también las columnas


def train_random_forest_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]
) -> RandomForestClassifier:
    log.info("--- Ajuste de Hiperparámetros para Random Forest (GridSearchCV) ---")

    param_grid_rf = parameters["param_grid_rf"]

    model_rf_base = RandomForestClassifier(random_state=parameters["random_state"])

    grid_search_rf = GridSearchCV(
        estimator=model_rf_base,
        param_grid=param_grid_rf,
        cv=parameters["grid_search_cv_folds"],
        scoring=parameters["grid_search_scoring"],
        n_jobs=parameters["grid_search_n_jobs"],
        verbose=parameters["grid_search_verbose"],
    )

    grid_search_rf.fit(X_train, y_train)

    log.info(
        f"\nMejores parámetros encontrados para Random Forest: {grid_search_rf.best_params_}"
    )
    log.info(
        f"Mejor AUC en validación cruzada para Random Forest: {grid_search_rf.best_score_:.4f}"
    )

    best_model_rf = grid_search_rf.best_estimator_
    log.info("--- Entrenamiento del Mejor Modelo Random Forest completado ---")

    return best_model_rf


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_columns: pd.Index,
) -> Dict[str, Any]:
    """
    Evalúa el modelo Random Forest entrenado y calcula métricas de rendimiento.

    Args:
        model: El modelo Random Forest entrenado.
        X_test: Características del conjunto de prueba.
        y_test: Variable objetivo del conjunto de prueba.
        X_train_columns: Nombres de las columnas de X_train (pd.Index).

    Returns:
        Un diccionario con métricas de evaluación, probabilidades de predicción y la importancia de las características.
    """
    log.info("\n--- Evaluación del Mejor Modelo Random Forest ---")

    y_pred_best_rf = model.predict(X_test)
    y_pred_proba_best_rf = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred_best_rf)
    classification_rep = classification_report(y_test, y_pred_best_rf, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba_best_rf)

    log.info(f"Precisión (Accuracy) del mejor Random Forest: {accuracy:.4f}")
    log.info("\nReporte de Clasificación del mejor Random Forest:")
    log.info(classification_report(y_test, y_pred_best_rf))
    log.info(f"AUC (Area Under the Curve) del mejor Random Forest: {auc_score:.4f}")

    # Importancia de las características
    importancia_caracteristicas_best_rf = pd.DataFrame(
        {"Caracteristica": X_train_columns, "Importancia": model.feature_importances_}
    ).sort_values(by="Importancia", ascending=False)

    log.info("\nImportancia de las Características del Mejor Random Forest:")
    log.info(importancia_caracteristicas_best_rf)

    evaluation_results = {
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "auc_score": auc_score,
        "y_pred_proba": y_pred_proba_best_rf.tolist(),
        "feature_importances": importancia_caracteristicas_best_rf.to_dict(
            orient="records"
        ),
    }
    log.info("--- Evaluación del modelo completada ---")
    return evaluation_results


def plot_roc_curve(
    y_test: pd.Series, evaluation_results: Dict[str, Any]
) -> matplotlib.figure.Figure:
    log.info("--- Generando Curva ROC ---")
    y_pred_proba_best_rf = np.array(evaluation_results["y_pred_proba"])
    auc_score_best_rf = evaluation_results["auc_score"]

    fpr_best_rf, tpr_best_rf, _ = roc_curve(y_test, y_pred_proba_best_rf)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr_best_rf,
        tpr_best_rf,
        color="darkgreen",
        lw=2,
        label=f"Curva ROC Mejor RF (AUC = {auc_score_best_rf:.2f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Clasificador Aleatorio",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curva ROC - Mejor Random Forest (GridSearchCV)")
    ax.legend(loc="lower right")
    ax.grid(True)
    log.info("--- Curva ROC generada ---")
    return fig


def plot_feature_importance(
    evaluation_results: Dict[str, Any],
) -> matplotlib.figure.Figure:
    log.info("--- Generando gráfico de Importancia de Características ---")
    importancia_df = pd.DataFrame(evaluation_results["feature_importances"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importancia", y="Caracteristica", data=importancia_df, ax=ax)
    ax.set_title("Importancia de las Características en el Mejor Random Forest")
    log.info("--- Gráfico de Importancia de Características generado ---")
    return fig
