"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

# Importa tus pipelines
from datawinekedro.pipelines import data_ingestion
from datawinekedro.pipelines import data_processing
from datawinekedro.pipelines import (
    model_training,
)  # Importa el nuevo módulo de pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Crea instancias de tus pipelines
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()
    model_training_pipeline = (
        model_training.create_pipeline()
    )  # Crea una instancia de la pipeline de modelado

    return {
        # La pipeline por defecto ejecutará ingesta, procesamiento y luego modelado en orden
        "__default__": data_ingestion_pipeline
        + data_processing_pipeline
        + model_training_pipeline,
        "data_ingestion": data_ingestion_pipeline,
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,  # Registra la pipeline de modelado
    }
