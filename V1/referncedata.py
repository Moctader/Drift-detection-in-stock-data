import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib  

def prepare_reference_dataset(df: pd.DataFrame):
    """Prepare reference dataset for monitoring."""
    target_col = "close"  # Use the same target as in training
    prediction_col = "predictions"
    
    features = ['open', 'high', 'low', 'volume', 'previous_close']
    
    # Convert datetime column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Split the data into reference data (excluding the last day) and current data (only the last day)
    last_day = df['datetime'].max().date()
    reference_data = df[df['datetime'].dt.date < last_day]
    current_data = df[df['datetime'].dt.date == last_day]
    
    # Prepare scoring data for reference and current data
    reference_scoring_data = reference_data[features]
    reference_target_data = reference_data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(reference_scoring_data, reference_target_data, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    MODEL_DIR = "models"
    Path(MODEL_DIR).mkdir(exist_ok=True)
    model_path = f"{MODEL_DIR}/linear_regression_model.joblib"
    joblib.dump(model, model_path)

    reference_data[prediction_col] = model.predict(reference_scoring_data)
    

    REFERENCE_DATA_DIR = "data/reference"
    Path(REFERENCE_DATA_DIR).mkdir(exist_ok=True)
    path = f"{REFERENCE_DATA_DIR}/reference_data.parquet"

    reference_data.to_parquet(path)
    current_data=load_model_and_predict(current_data)
    return reference_data, current_data  # Return both reference data and current data



def load_model_and_predict(current_data: pd.DataFrame):
    """Load the saved model and make predictions on the current data."""
    prediction_col = "predictions"
    features = ['open', 'high', 'low', 'volume', 'previous_close']
    
    # Load the trained model
    MODEL_DIR = "models"
    model_path = f"{MODEL_DIR}/linear_regression_model.joblib"
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Prepare scoring data for current data
    current_scoring_data = current_data[features]

    # Generate predictions for current data
    current_data[prediction_col] = model.predict(current_scoring_data)
    

    return current_data