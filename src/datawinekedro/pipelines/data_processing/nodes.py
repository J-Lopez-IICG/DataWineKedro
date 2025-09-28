import pandas as pd
from typing import Dict, Any


def passthrough_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Una función simple que recibe datos y los devuelve sin cambios.
    Kedro se encarga de cargar desde CSV y guardar en Excel según el catálogo.
    """
    return data
