from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data,
    train_random_forest_model,
    train_decision_tree_model,
    train_xgboost_model,
    evaluate_model,
    compare_models_and_select_best,
    aggregate_models,
    aggregate_metrics,
    plot_roc_curve,
    plot_feature_importance,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea la pipeline de entrenamiento, evaluación y selección del mejor modelo.

    Returns:
        Un objeto Kedro Pipeline.
    """
    return pipeline(
        [
            # Nodo 1: Divide los datos en conjuntos de entrenamiento y prueba
            node(
                func=split_data,
                inputs=["model_input_data", "params:model_options"],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "X_train_columns",
                ],
                name="split_data_node",
            ),
            # --- Rama para Random Forest ---
            node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="rf_model",
                name="train_rf_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["rf_model", "X_test", "y_test", "X_train_columns"],
                outputs="rf_metrics",
                name="evaluate_rf_model_node",
            ),
            # --- Rama para Decision Tree ---
            node(
                func=train_decision_tree_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="dt_model",
                name="train_dt_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["dt_model", "X_test", "y_test", "X_train_columns"],
                outputs="dt_metrics",
                name="evaluate_dt_model_node",
            ),
            # --- Rama para XGBoost ---
            node(
                func=train_xgboost_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="xgb_model",
                name="train_xgb_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["xgb_model", "X_test", "y_test", "X_train_columns"],
                outputs="xgb_metrics",
                name="evaluate_xgb_model_node",
            ),
            # --- Nodos de Agregación ---
            node(
                func=aggregate_models,
                inputs={
                    "rf_model": "rf_model",
                    "dt_model": "dt_model",
                    "xgb_model": "xgb_model",
                },
                outputs="all_trained_models",
                name="aggregate_models_node",
            ),
            node(
                func=aggregate_metrics,
                inputs={
                    "rf_metrics": "rf_metrics",
                    "dt_metrics": "dt_metrics",
                    "xgb_metrics": "xgb_metrics",
                },
                outputs="all_evaluation_metrics",
                name="aggregate_metrics_node",
            ),
            # --- Nodo de Comparación ---
            node(
                func=compare_models_and_select_best,
                inputs={
                    "models": "all_trained_models",
                    "metrics": "all_evaluation_metrics",
                },
                outputs=[
                    "best_model",
                    "best_model_metrics",
                ],  # El nodo ahora emite dos outputs
                name="select_best_model_node",
            ),
            # --- Nodos de gráficos para el mejor modelo ---
            node(
                func=plot_roc_curve,
                inputs=["y_test", "best_model_metrics"],
                outputs="best_model_roc_curve_plot",
                name="plot_best_model_roc_curve_node",
            ),
            node(
                func=plot_feature_importance,
                inputs="best_model_metrics",
                outputs="best_model_feature_importance_plot",
                name="plot_best_model_feature_importance_node",
            ),
        ],
        namespace="model_training",
        inputs="model_input_data",
        outputs={
            "best_model",
            "best_model_metrics",
            "best_model_roc_curve_plot",
            "best_model_feature_importance_plot",
        },
    )
