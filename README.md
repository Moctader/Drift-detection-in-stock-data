# Data Drift and Model Performance Monitoring

This project demonstrates how to monitor data drift and model performance using Evidently AI and a PostgreSQL database. The scripts read intraday stock data from a PostgreSQL database, process the data in chunks, and generate reports on data drift and model performance.

## Prerequisites

- Python 3.11
- PostgreSQL database with intraday stock data
- Required Python packages (see `requirements.txt`)

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Moctader/drift-detection-in-stock-data.git
    cd drift-detection-in-stock-data

    ```

2. **Create and activate a virtual environment**:

    ```sh
    python -m venv env
    source env/bin/activate
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Start Docker containers**:

    ```sh
    docker-compose up -d
    ```

5. **Run the data fetching script** in one terminal:

    ```sh
    python dynamic_intraday_data_fetcher.py
    ```

6. **Run the drift detection script** in another terminal:

    ```sh
    python Evidently-drift-detection.py
    ```

# Script Overview

The scripts handle and analyze intraday stock data, providing visualizations and reports to assess data quality and model performance.

## Functions

- **`read_data()`**: Reads data from the PostgreSQL database.
- **`plot_distributions(reference_data, current_data)`**: Plots line charts for reference and current data.
- **`bar_plot_distributions(reference_data, current_data)`**: Plots bar charts for reference and current data.
- **`get_dataset_drift_report(reference, current, column_mapping)`**: Generates a data drift report comparing reference and current datasets.
- **`get_model_performance_report(reference, current, column_mapping)`**: Generates a model performance report comparing reference and current datasets.
- **`detect_dataset_drift(report)`**: Detects dataset drift based on the generated report.
- **`process_data_in_chunks(df)`**: Processes the data in chunks of 10,000 records and generates reports.

## Main Workflow

1. **Read Data**: The script reads intraday stock data from the PostgreSQL database.
2. **Process Data in Chunks**: The data is processed in chunks of 10,000 records.
3. **Generate Reports**: For each chunk, the script generates data drift and model performance reports.
4. **Open Reports**: The generated reports are saved as HTML files and opened in the default web browser.
