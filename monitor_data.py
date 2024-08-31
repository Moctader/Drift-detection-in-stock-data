import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Text

import pandas as pd
import pendulum
from prefect import flow, task
from data_quality import commit_data_metrics_to_db
from utils import extract_batch_data, get_batch_interval

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetSummaryMetric
from evidently.report import Report

@task
def generate_reports(
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    num_features: List[Text],
    cat_features: List[Text],
    prediction_col: Text,
    timestamp: float,
) -> None:
    """
    Generate data quality and data drift reports and
    commit metrics to the database.

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

    logging.info("Prepare column mapping")
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = num_features
    column_mapping.prediction = prediction_col

    if current_data.predictions.notnull().sum() > 0:
        column_mapping.prediction = prediction_col

    logging.info("Data quality report")
    data_quality_report = Report(metrics=[DatasetSummaryMetric()])
    data_quality_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    logging.info("Data drift report")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    data_drift_report.save_html("1-data_drift_report.html")

    logging.info("Commit metrics into database")
    data_quality_report_content: Dict = data_quality_report.as_dict()
    data_drift_report_content: Dict = data_drift_report.as_dict()
    commit_data_metrics_to_db(
        data_quality_report=data_quality_report_content,
        data_drift_report=data_drift_report_content,
        timestamp=timestamp,
    )

@flow(flow_run_name="monitor-data-on-{ts}", log_prints=True)
def monitor_data(current_data, ts) -> None:
    """Build and save data validation reports.

    Args:
        ts (str): ISO 8601 formatted timestamp.
    """
    num_features = ['datetime','open', 'high', 'low', 'close', 'volume', 'previous_close']
    cat_features = []
    prediction_col = "predictions"

    # Prepare reference data
    DATA_REF_DIR = "data/reference"
    ref_path = f"{DATA_REF_DIR}/reference_data.parquet"
    ref_data = pd.read_parquet(ref_path)
    columns: List[Text] = num_features + cat_features + [prediction_col]
    reference_data = ref_data.loc[:, columns]

    if current_data.shape[0] == 0:
        # Skip monitoring if current data is empty
        # Usually it may happen for few first batches
        print("Current data is empty!")
        print("Skip model monitoring")
    else:
        # Prepare column_mapping object
        # for Evidently reports and generate reports
        generate_reports(
            current_data=current_data,
            reference_data=reference_data,
            num_features=num_features,
            cat_features=cat_features,
            prediction_col=prediction_col,
            timestamp=pendulum.parse(ts).timestamp(),
        )