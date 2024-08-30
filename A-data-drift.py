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
from func import process_time_series_data, detect_drift


app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database connection parameters
DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/your_database"

from sqlalchemy import create_engine
import pandas as pd
import logging

DATABASE_URL = "postgresql://your_user:your_password@localhost:5432/your_database"

def read_data() -> pd.DataFrame:
    """Read data from PostgreSQL using SQLAlchemy and return only specified numerical features."""
    try:
        # Create a database engine
        engine = create_engine(DATABASE_URL)
        
        # Define numerical features to be selected
        numerical_features = ['open', 'high', 'low', 'close', 'volume', 'previous_close']
        
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
        
        return df

    except Exception as e:
        logging.error(f"Error reading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there's an error






@app.get('/detect-drift')
def detect_drift_endpoint(window_size: int = 3000) -> FileResponse:
    logging.info('Read data from database')

    # Read data from the database
    df = read_data()
    if df.empty:
        return {"error": "No data found or an error occurred while reading the data."}

    # Prepare data for drift detection
    try:
        #prepare_data_for_drift_detection(df, window_size)
        #reference_data, current_data = 
        reference, target=process_time_series_data(df, window_size=3000)
        detect_drift(reference, target)
    except ValueError as e:
        logging.error(f"Error preparing data: {e}")
        return {"error": str(e)}

    # Detect drift

    logging.info('Drift detection completed and results saved.')
    report_path=f"drift_detection_report.html"
    # Return the file as a response
    return FileResponse(report_path)

# Run the application with Uvicorn (optional, usually done from command line)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


