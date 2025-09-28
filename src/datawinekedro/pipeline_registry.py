"""Project pipelines."""

from typing import Dict

# Elimina la siguiente línea:
# from kedro.framework.project import register_pipelines
from kedro.pipeline import Pipeline

# Importa tus pipelines
from datawinekedro.pipelines import data_processing
from datawinekedro.pipelines import data_ingestion


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Crea instancias de tus pipelines
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()

    return {
        # La pipeline por defecto ejecutará primero la ingesta y luego el procesamiento
        "__default__": data_ingestion_pipeline + data_processing_pipeline,
        "data_ingestion": data_ingestion_pipeline,  # Puedes ejecutar solo la ingesta
        "data_processing": data_processing_pipeline,  # Puedes ejecutar solo el procesamiento
    }
