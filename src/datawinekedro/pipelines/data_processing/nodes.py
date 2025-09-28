import pandas as pd
from typing import Dict, Any


import pandas as pd
from pandas.api.types import is_numeric_dtype
import logging

# Obtener el logger de Kedro para registrar mensajes
log = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza los pasos de limpieza de datos: maneja valores nulos y duplicados.
    Registra el estado de los datos antes y después de la limpieza.

    Args:
        data: El DataFrame de entrada.

    Returns:
        El DataFrame limpio.
    """
    log.info("--- Iniciando limpieza de datos ---")

    # 1. Revisar si hay valores nulos en cada columna
    log.info("Valores nulos por columna antes de la limpieza:")
    log.info(data.isnull().sum())

    # 2. Eliminar filas que contengan al menos un valor nulo
    # Usamos .copy() para evitar SettingWithCopyWarning en operaciones posteriores
    data_cleaned = data.dropna().copy()

    log.info("\nValores nulos por columna después de la limpieza:")
    log.info(data_cleaned.isnull().sum())

    # 3. Revisar si hay filas duplicadas
    log.info(
        f"\nNúmero de filas duplicadas encontradas: {data_cleaned.duplicated().sum()}"
    )

    # 4. Eliminar filas duplicadas
    data_cleaned.drop_duplicates(inplace=True)

    log.info(
        f"Número de filas duplicadas después de la limpieza: {data_cleaned.duplicated().sum()}"
    )

    # 5. Mostrar información general del DataFrame limpio
    log.info("\nInformación del DataFrame después de la limpieza:")
    data_cleaned.info()

    # 6. Verificar que todas las columnas sean numéricas (int o float)
    log.info("\n--- Verificación de Tipos de Datos ---")
    non_numeric_columns = []
    for column in data_cleaned.columns:
        if not is_numeric_dtype(data_cleaned[column]):
            non_numeric_columns.append(column)

    if not non_numeric_columns:
        log.info(
            "¡Excelente! Todas las columnas tienen un formato numérico (int o float)."
        )
    else:
        log.warning("¡Atención! Se encontraron columnas con formato no numérico:")
        for col in non_numeric_columns:
            log.warning(f"- Columna: '{col}', Tipo: {data_cleaned[col].dtype}")

    log.info("--- Limpieza de datos completada ---")
    return data_cleaned


def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas del DataFrame a español, minúsculas con guiones bajos.
    Esto mejora la legibilidad y consistencia.

    Args:
        data: El DataFrame de entrada con los nombres de columna originales.

    Returns:
        El DataFrame con las columnas renombradas.
    """
    log.info("--- Iniciando renombrado de columnas ---")
    column_mapping = {
        "fixed acidity": "Acidez_Fija",
        "volatile acidity": "Acidez_Volatil",
        "citric acid": "Acido_Citrico",
        "residual sugar": "Azucar_Residual",
        "chlorides": "Cloruros",
        "free sulfur dioxide": "Dioxido_Azufre_Libre",
        "total sulfur dioxide": "Dioxido_Azufre_Total",
        "density": "Densidad",
        "pH": "ph",
        "sulphates": "Sulfatos",
        "alcohol": "Alcohol",
        "quality": "Calidad",
    }

    data_renamed = data.rename(columns=column_mapping).copy()

    log.info("Nombres de las columnas actualizados:")
    log.info(data_renamed.columns.tolist())
    log.info("--- Renombrado de columnas completado ---")
    return data_renamed


def create_target_and_select_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo binaria 'Calidad_Binaria' (1 si Calidad >= 7, 0 en otro caso)
    y selecciona las características finales que se usarán para el modelado.

    Args:
        data: El DataFrame con las columnas renombradas.

    Returns:
        Un DataFrame con las características seleccionadas y la nueva variable objetivo binaria.
    """
    log.info(
        "--- Iniciando creación de variable objetivo y selección de características ---"
    )

    # Crear la nueva columna 'Calidad_Binaria'
    data_processed = data.copy()  # Trabajar en una copia para no modificar el original
    data_processed["Calidad_Binaria"] = (data_processed["Calidad"] >= 7).astype(int)

    # Verificar la distribución de la nueva columna
    log.info("Distribución de la nueva columna 'Calidad_Binaria':")
    log.info(data_processed["Calidad_Binaria"].value_counts())

    # Definir las columnas finales que queremos mantener para el modelo
    columnas_a_mantener = [
        "Acidez_Fija",
        "Acidez_Volatil",
        "Acido_Citrico",
        "Azucar_Residual",
        "Cloruros",
        "Dioxido_Azufre_Libre",
        "Dioxido_Azufre_Total",
        "Densidad",
        "ph",
        "Sulfatos",
        "Alcohol",
        "Calidad_Binaria",  # Esta es nuestra variable objetivo
    ]
    data_seleccionada = data_processed[columnas_a_mantener].copy()

    log.info("DataFrame con las columnas seleccionadas (primeras 5 filas):")
    log.info(data_seleccionada.head())

    log.info("\nInformación del DataFrame con columnas seleccionadas:")
    data_seleccionada.info()

    log.info(
        "--- Creación de variable objetivo y selección de características completada ---"
    )
    return data_seleccionada
