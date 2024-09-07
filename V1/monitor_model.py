import argparse
import logging
from typing import Dict
from typing import List
from typing import Text

import pandas as pd
import pendulum
from prefect import flow
from prefect import task
from model_performance import commit_model_metrics_to_db
#from src.pipelines.monitor_data import prepare_current_data
from utils import get_batch_interval

from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import RegressionQualityMetric
from evidently.report import Report


@task(name="prepare_current_data")
def generate_reports(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    num_features: List[Text],
    cat_features: List[Text],
    prediction_col: Text,
    target_col: Text,
    timestamp: float,
) -> None:
    """
    Generate data quality and data drift reports
    and commit metrics to the database.

    Args:
        current_data (pd.DataFrame):
            The current DataFrame with features and predictions.
        reference_data (pd.DataFrame):
            The reference DataFrame with features and predictions.
        num_features (List[Text]):
            List of numerical feature column names.
        cat_features (List[Text]):
            List of categorical feature column names.
        prediction_col (Text):
            Name of the prediction column.
        timestamp (float):
            Metric pipeline execution timestamp.
    """

    print("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col
    column_mapping.numerical_features = num_features
    column_mapping.categorical_features = cat_features

    logging.info("Create a model performance report")
    model_performance_report = Report(metrics=[RegressionQualityMetric()])
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    model_performance_report.save_html("model-performance/model_performance_report.html")

    logging.info("Target drift report")
    target_drift_report = Report(metrics=[ColumnDriftMetric(target_col)])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    target_drift_report.save_html("Target-drift/Target-drift.html")


    logging.info("Save metrics to database")
    model_performance_report_content: Dict = model_performance_report.as_dict()
    target_drift_report_content: Dict = target_drift_report.as_dict()
    commit_model_metrics_to_db(
        model_performance_report=model_performance_report_content,
        target_drift_report=target_drift_report_content,
        timestamp=timestamp,
    )


@flow(flow_run_name="monitor-model-on-{ts}", log_prints=True)
def monitor_model(current_data, ts) -> None:
    """Build and save monitoring reports.

    Args:
        ts (pendulum.DateTime, optional): Timestamp.
        interval (int, optional): Interval. Defaults to 60.
    """

    DATA_REF_DIR = "data/reference"
    target_col = "close"
    num_features = ['datetime','open', 'high', 'low', 'close', 'volume', 'previous_close']
    cat_features = []
    prediction_col = "predictions"

    if current_data.shape[0] == 0:
        # Skip monitoring if current data is empty
        # Usually it may happen for few first batches
        print("Current data is empty!")
        print("Skip model monitoring")

    else:

        # Prepare reference data
        ref_path = f"{DATA_REF_DIR}/reference_data.parquet"
        ref_data = pd.read_parquet(ref_path)
        columns: List[Text] = num_features + cat_features + [prediction_col]
        reference_data = ref_data.loc[:, columns]
        # Generate reports
        generate_reports(
            current_data=current_data,
            reference_data=reference_data,
            num_features=num_features,
            cat_features=cat_features,
            prediction_col=prediction_col,
            target_col=target_col,
            timestamp=pendulum.parse(ts).timestamp(),
        )


