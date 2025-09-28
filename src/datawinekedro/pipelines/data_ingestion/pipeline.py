from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea la pipeline de ingesta de datos.
    Carga el CSV raw y lo guarda como el Excel intermedio.

    Returns:
        Un objeto Kedro Pipeline.
    """
    return pipeline(
        [
            node(
                func=load_raw_data,
                inputs="red_wine_csv",  # Lee el CSV raw
                outputs="processed_wine_data",  # Guarda como Excel (según catalog.yml)
                name="load_raw_csv_to_excel_node",
            ),
        ],
        namespace="data_ingestion",
        inputs="red_wine_csv",
        outputs="processed_wine_data",
    )
