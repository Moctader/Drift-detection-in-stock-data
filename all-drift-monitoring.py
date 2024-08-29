from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import logging
from typing import Text
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset, TargetDriftPreset
from sqlalchemy import create_engine
from sklearn import ensemble
import numpy as np
import uvicorn

app = FastAPI()

# Database connection parameters
DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/your_database"

def read_data():
    """Read data from PostgreSQL using SQLAlchemy."""
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM intraday_data"
    df = pd.read_sql_query(query, engine)
    return df

def get_column_mapping(**kwargs) -> ColumnMapping:
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.numerical_features = ['open', 'high', 'low', 'volume', 'previous_close']
    column_mapping.categorical_features = []
    return column_mapping

def get_dataset_drift_report(reference: pd.DataFrame, current: pd.DataFrame, column_mapping: ColumnMapping):
    """Returns a data drift report."""
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    return data_drift_report

def detect_dataset_drift(report: Report):
    """Detect dataset drift from the report."""
    return report.as_dict()["metrics"][0]["result"]["dataset_drift"]

def get_model_performance_report(reference_data, current_data, column_mapping):
    """Generate model performance report."""
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    return report

def process_data_in_chunks(df):
    """Process the data in chunks of 10,000 records."""
    chunk_size = 10000
    num_chunks = len(df) // chunk_size

    for i in range(num_chunks - 1):
        start_idx_ref = i * chunk_size
        end_idx_ref = start_idx_ref + chunk_size
        start_idx_cur = end_idx_ref
        end_idx_cur = start_idx_cur + chunk_size

        # Reference data
        reference_data = df.iloc[start_idx_ref:end_idx_ref]

        # Current data
        current_data = df.iloc[start_idx_cur:end_idx_cur]

        # Perform drift detection and model performance evaluation
        target = 'close'
        prediction = 'prediction'
        numerical_features = ['open', 'high', 'low', 'volume', 'previous_close']
        categorical_features = []

        regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
        regressor.fit(reference_data[numerical_features + categorical_features], reference_data[target])
        ref_prediction = regressor.predict(reference_data[numerical_features + categorical_features])
        current_prediction = regressor.predict(current_data[numerical_features + categorical_features])
        
        reference_data['prediction'] = ref_prediction
        current_data['prediction'] = current_prediction

        column_mapping = ColumnMapping()
        column_mapping.target = target
        column_mapping.prediction = prediction
        column_mapping.numerical_features = numerical_features
        column_mapping.categorical_features = categorical_features

        # Generate and detect dataset drift report
        data_drift_report = get_dataset_drift_report(reference_data, current_data, column_mapping)
        drift_detected = detect_dataset_drift(data_drift_report)
    
        if drift_detected:
            print("Dataset drift detected.")
        else:
            print("No dataset drift detected.")

        # Generate and save the dataset drift report
        data_drift_report.save_html(f"data_drift_report_chunk_{i + 1}_to_{i + 2}.html")

        # Generate and save the model performance report
        model_performance_report = get_model_performance_report(reference_data, current_data, column_mapping)
        model_performance_report.save_html(f"model_performance_report_chunk_{i + 1}_to_{i + 2}.html")

        return current_data, reference_data

def build_target_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping
) -> Text:
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    report_path = 'target_drift.html'
    target_drift_report.save_html(report_path)
    return report_path

@app.get('/detect-drift')
def detect_drift(window_size: int = 3000) -> FileResponse:
    logging.info('Read data from database')
    df = read_data()
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    logging.info('Process data in chunks for drift detection')
    current_data, reference_data = process_data_in_chunks(df)
    
    # Assuming the last chunk's report is the one to return
    report_path1 = "data_drift_report_chunk_{}_to_{}.html".format(len(df) // 10000 - 1, len(df) // 10000)
    
    logging.info('Return report as html')
    return FileResponse(report_path1)

@app.get('/monitor-model')
def monitor_model_performance(window_size: int = 3000) -> FileResponse:
    logging.info('Read data from database')
    df = read_data()
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    logging.info('Process data in chunks')
    current_data, reference_data = process_data_in_chunks(df)
    
    # Assuming the last chunk's report is the one to return
    report_path = "model_performance_report_chunk_{}_to_{}.html".format(len(df) // 10000 - 1, len(df) // 10000)
    
    logging.info('Return report as html')
    return FileResponse(report_path)

@app.get('/monitor-target')
def monitor_target_drift(window_size: int = 3000) -> FileResponse:
    logging.info('Read data from database')
    df = read_data()
    
    # Ensure 'datetime' is a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    logging.info('Process data in chunks')
    current_data, reference_data = process_data_in_chunks(df)

    logging.info('Build report')
    column_mapping: ColumnMapping = get_column_mapping()
    report_path: Text = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )

    logging.info('Return report as html')
    return FileResponse(report_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)