from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd
import logging
from typing import Text
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset, TargetDriftPreset
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import numpy as np
import uvicorn
from func import process_time_series_data, detect_data_drift, model_drift_detection
from train import train_model
from referncedata import prepare_reference_dataset
from monitor_data import monitor_data
import pendulum
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'H' is deprecated and will be removed in a future version, please use 'h' instead.")


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database connection parameters
DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/your_database"

def read_data() -> pd.DataFrame:
    """Read data from PostgreSQL using SQLAlchemy and return only specified numerical features."""
    try:
        # Create a database engine
        engine = create_engine(DATABASE_URL)
        
        # Define numerical features to be selected
        numerical_features = ['datetime','open', 'high', 'low', 'close', 'volume', 'previous_close']
        
        # Read the data from PostgreSQL
        query = "SELECT * FROM intraday_data"
        df = pd.read_sql_query(query, engine)
        
        # Check if all numerical features exist in the DataFrame
        missing_features = [col for col in numerical_features if col not in df.columns]
        if missing_features:
            logging.error(f"Missing columns in the data: {', '.join(missing_features)}")
            return pd.DataFrame()  # Return empty DataFrame if columns are missing
        
        # Select only the specified numerical features
        df = df[numerical_features]
        
        # Normalize the numerical features
        for col in ['open', 'high', 'low', 'close', 'volume', 'previous_close']:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        return df

    except Exception as e:
        logging.error(f"Error reading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there's an error


if __name__ == "__main__":
    df = read_data()
    train_model(df)
    reference_data, current_data = prepare_reference_dataset(df)
    ts = pendulum.now()
    
    # Pass the ISO 8601 formatted timestamp to monitor_data
    monitor_data(current_data, ts.to_iso8601_string())