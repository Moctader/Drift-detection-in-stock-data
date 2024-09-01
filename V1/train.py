import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

def train_model(df: pd.DataFrame) -> LinearRegression:
    """Train a linear regression model on the specified DataFrame."""
    # Define the features and target
    features = ['open', 'high', 'low', 'volume', 'previous_close']
    target = 'close'
    
    # Split the data into features and target
    X = df[features]
    y = df[target]
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Get predictions for training and validation sets
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)


    joblib.dump(model, "models/model.joblib")
    
    return model