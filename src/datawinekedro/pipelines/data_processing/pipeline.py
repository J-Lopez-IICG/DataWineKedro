from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_data, rename_columns, create_target_and_select_features


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea la pipeline de procesamiento de datos.
    Define el flujo de trabajo desde los datos intermedios hasta los datos listos para el modelo.

    Returns:
        Un objeto Kedro Pipeline.
    """
    return pipeline(
        [
            # Nodo 1: Limpia los datos (maneja nulos y duplicados)
            node(
                func=clean_data,
                inputs="processed_wine_data",  # Lee los datos intermedios (Excel)
                outputs="cleaned_wine_data",  # Guarda los datos limpios (Excel)
                name="clean_wine_data_node",
            ),
            # Nodo 2: Renombra las columnas para mayor claridad
            node(
                func=rename_columns,
                inputs="cleaned_wine_data",  # Toma los datos limpios
                outputs="renamed_wine_data",  # Guarda los datos con columnas renombradas (Excel)
                name="rename_wine_columns_node",
            ),
            # Nodo 3: Crea la variable objetivo y selecciona las características finales
            node(
                func=create_target_and_select_features,
                inputs="renamed_wine_data",  # Toma los datos con columnas renombradas
                outputs="model_input_data",  # Guarda los datos finales listos para el modelado (CSV)
                name="create_target_and_select_features_node",
            ),
        ],
        namespace="data_processing",  # Agrupa estos nodos bajo un namespace para mejor organización
        inputs="processed_wine_data",  # Define la entrada principal de esta pipeline
        outputs="model_input_data",  # Define la salida principal de esta pipeline
    )
