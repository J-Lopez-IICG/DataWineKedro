from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_data,
    train_random_forest_model,
    evaluate_model,
    plot_roc_curve,
    plot_feature_importance,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea la pipeline de entrenamiento y evaluación del modelo.

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
                ],  # Añadido "X_train_columns"
                name="split_data_node",
            ),
            # Nodo 2: Entrena el modelo Random Forest con GridSearchCV
            node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="best_random_forest_model",
                name="train_random_forest_model_node",
            ),
            # Nodo 3: Evalúa el modelo y calcula métricas
            node(
                func=evaluate_model,
                inputs=[
                    "best_random_forest_model",
                    "X_test",
                    "y_test",
                    "X_train_columns",  # Ahora usa el nuevo dataset "X_train_columns"
                ],
                outputs="rf_evaluation_metrics",
                name="evaluate_random_forest_model_node",
            ),
            # Nodo 4: Genera el gráfico de la curva ROC
            node(
                func=plot_roc_curve,
                inputs=["y_test", "rf_evaluation_metrics"],
                outputs="rf_roc_curve_plot",
                name="plot_rf_roc_curve_node",
            ),
            # Nodo 5: Genera el gráfico de importancia de características
            node(
                func=plot_feature_importance,
                inputs="rf_evaluation_metrics",
                outputs="rf_feature_importance_plot",
                name="plot_rf_feature_importance_node",
            ),
        ],
        namespace="model_training",
        inputs="model_input_data",
        outputs={
            "best_random_forest_model",
            "rf_evaluation_metrics",
            "rf_roc_curve_plot",
            "rf_feature_importance_plot",
        },
    )
