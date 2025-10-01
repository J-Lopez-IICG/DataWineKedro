import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.figure

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
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


def train_decision_tree_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]
) -> DecisionTreeClassifier:
    log.info("--- Ajuste de Hiperparámetros para Decision Tree (GridSearchCV) ---")

    param_grid_dt = parameters["param_grid_dt"]

    model_dt_base = DecisionTreeClassifier(random_state=parameters["random_state"])

    grid_search_dt = GridSearchCV(
        estimator=model_dt_base,
        param_grid=param_grid_dt,
        cv=parameters["grid_search_cv_folds"],
        scoring=parameters["grid_search_scoring"],
        n_jobs=parameters["grid_search_n_jobs"],
        verbose=parameters["grid_search_verbose"],
    )

    grid_search_dt.fit(X_train, y_train)

    log.info(
        f"\nMejores parámetros encontrados para Decision Tree: {grid_search_dt.best_params_}"
    )
    log.info(
        f"Mejor AUC en validación cruzada para Decision Tree: {grid_search_dt.best_score_:.4f}"
    )

    best_model_dt = grid_search_dt.best_estimator_
    log.info("--- Entrenamiento del Mejor Modelo Decision Tree completado ---")

    return best_model_dt


def train_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]
) -> xgb.XGBClassifier:
    log.info("--- Ajuste de Hiperparámetros para XGBoost (GridSearchCV) ---")

    param_grid_xgb = parameters["param_grid_xgb"]

    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count

    model_xgb_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=parameters["random_state"],
        scale_pos_weight=scale_pos_weight_value,
    )

    grid_search_xgb = GridSearchCV(
        estimator=model_xgb_base,
        param_grid=param_grid_xgb,
        cv=parameters["grid_search_cv_folds"],
        scoring=parameters["grid_search_scoring"],
        n_jobs=parameters["grid_search_n_jobs"],
        verbose=parameters["grid_search_verbose"],
    )

    grid_search_xgb.fit(X_train, y_train)

    log.info(
        f"\nMejores parámetros encontrados para XGBoost: {grid_search_xgb.best_params_}"
    )
    log.info(
        f"Mejor AUC en validación cruzada para XGBoost: {grid_search_xgb.best_score_:.4f}"
    )

    best_model_xgb = grid_search_xgb.best_estimator_
    log.info("--- Entrenamiento del Mejor Modelo XGBoost completado ---")

    return best_model_xgb


def compare_models_and_select_best(
    models: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Compara los modelos basándose en su métrica AUC y devuelve el mejor modelo y sus métricas.
    """
    best_model_name = None
    best_auc = -1.0
    best_metrics_dict = {}

    for model_key, metric_dict in metrics.items():
        auc = metric_dict.get("auc_score", 0.0)
        log.info(f"Evaluando modelo: {model_key} con AUC: {auc}")
        if auc > best_auc:
            best_auc = auc
            best_model_name = model_key  # Esto será 'rf_metrics', 'dt_metrics', etc.
            best_metrics_dict = metric_dict

    if best_model_name is None:
        raise ValueError("No se pudo determinar el mejor modelo.")

    # Extraer el prefijo del nombre del modelo (ej. 'rf' de 'rf_metrics')
    model_prefix = best_model_name.replace("_metrics", "")
    # Construir la clave correcta para el diccionario de modelos (ej. 'rf_model')
    model_key_in_models = f"{model_prefix}_model"

    log.info(f"El mejor modelo es '{model_key_in_models}' con un AUC de {best_auc:.4f}")

    return models[model_key_in_models], best_metrics_dict


def aggregate_models(**kwargs) -> Dict[str, Any]:
    """Agrega todos los modelos entrenados en un solo diccionario.
    Las claves del diccionario serán los nombres de los datasets de entrada.
    """
    return kwargs


def aggregate_metrics(**kwargs) -> Dict[str, Dict[str, Any]]:
    """Agrega todas las métricas de evaluación en un solo diccionario.
    Las claves del diccionario serán los nombres de los datasets de entrada.
    """
    return kwargs


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train_columns: pd.Index,
) -> Dict[str, Any]:
    """
    Evalúa el modelo entrenado y calcula métricas de rendimiento.

    Args:
        model: El modelo entrenado.
        X_test: Características del conjunto de prueba.
        y_test: Variable objetivo del conjunto de prueba.
        X_train_columns: Nombres de las columnas de X_train (pd.Index).

    Returns:
        Un diccionario con métricas de evaluación, probabilidades de predicción y la importancia de las características.
    """
    model_name = model.__class__.__name__
    log.info(f"\n--- Evaluación del Modelo {model_name} ---")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    log.info(f"Precisión (Accuracy) de {model_name}: {accuracy:.4f}")
    log.info(f"\nReporte de Clasificación de {model_name}:")
    log.info(classification_report(y_test, y_pred))
    log.info(f"AUC (Area Under the Curve) de {model_name}: {auc_score:.4f}")

    # Importancia de las características
    importancia_caracteristicas = pd.DataFrame(
        {"Caracteristica": X_train_columns, "Importancia": model.feature_importances_}
    ).sort_values(by="Importancia", ascending=False)

    log.info(f"\nImportancia de las Características de {model_name}:")
    log.info(importancia_caracteristicas)

    evaluation_results = {
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "auc_score": auc_score,
        "y_pred_proba": y_pred_proba.tolist(),
        "feature_importances": importancia_caracteristicas.to_dict(orient="records"),
    }
    log.info(f"--- Evaluación del modelo {model_name} completada ---")
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
