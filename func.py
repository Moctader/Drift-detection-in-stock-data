import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import logging
from typing import Text
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset, TargetDriftPreset
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
import numpy as np

def process_time_series_data(df: pd.DataFrame, window_size: int, split_ratio: float = 0.7) -> pd.DataFrame:
    """
    Process time series data with the following steps:
    1. Data Cleaning
    2. Detrending
    3. Normalization/Standardization
    4. Windowing and Feature Engineering
    5. Splitting the Data into Reference and Current Periods

    Args:
    df (pd.DataFrame): The input DataFrame containing time series data.
    window_size (int): Size of the rolling window for feature extraction.
    split_ratio (float): Ratio to split the data into reference and current periods.

    Returns:
    pd.DataFrame: Processed DataFrame with time windows.
    """
    
    # Step 1: Data Cleaning
    # Handle missing values by filling forward
    df = df.fillna(method='ffill')
    
    # Detect and handle outliers using z-score method
    from scipy import stats
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]  # Keep rows with z-scores < 3
    
    # Step 2: Detrending
    numerical_features = ['open', 'high', 'low', 'close', 'volume', 'previous_close']
    for feature in numerical_features:
        df[feature] = detrend(df[feature].values)
    
    # Step 3: Normalization/Standardization
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # # Step 4: Windowing and Feature Engineering
    # def create_windows(df, window_size):
    #     windows = []
    #     for start in range(len(df) - window_size + 1):
    #         end = start + window_size
    #         window = df.iloc[start:end].copy()
    #         window['window_start'] = start
    #         window['window_end'] = end
    #         windows.append(window)
    #     return pd.concat(windows, ignore_index=True)
    
    # df_windows = create_windows(df, window_size)
    
    # Step 5: Splitting the Data into Reference and Current Periods
    split_index = int(len(df) * split_ratio)
    reference_data = df[:split_index]
    current_data = df[split_index:]
    print("Reference Data:")
    print(reference_data.shape)
    print("Current Data:")
    print(current_data.shape)
    
    # Return both reference and current data for further processing
    return reference_data, current_data



def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect drift using Evidently library and return the results.

    Here are the names of the statistical tests available in evidently for data drift detection:

    psi - Population Stability Index
    ks - Kolmogorov-Smirnov Test
    chi2 - Chi-Squared Test
    cvm - Cramér-Von Mises Test
    """
    # Define ColumnMapping
    column_mapping = ColumnMapping()
    
    # Create the report
    report = Report(metrics=[DataDriftPreset(stattest='ks', stattest_threshold='-3')])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report_file=report.save_html("drift_detection_report.html")
    
    # Save the report to a CSV file
    
    #return report_file