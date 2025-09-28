import pandas as pd
import logging

log = logging.getLogger(__name__)


def load_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Carga los datos raw (CSV) y los devuelve.
    Kedro se encargará de guardarlos en el formato especificado en el catálogo (Excel).
    """
    log.info("--- Iniciando carga de datos raw ---")
    log.info(f"Datos raw cargados: {len(data)} filas, {len(data.columns)} columnas.")
    log.info("--- Carga de datos raw completada ---")
    return data
