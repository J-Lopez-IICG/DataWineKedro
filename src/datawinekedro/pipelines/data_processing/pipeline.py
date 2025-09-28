"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import passthrough_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=passthrough_data,
                inputs="red_wine_csv",
                outputs="processed_wine_data",
                name="convert_csv_to_xlsx_node",
            )
        ]
    )
