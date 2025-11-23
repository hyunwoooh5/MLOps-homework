"""
This script calculates data drift and other metrics for a machine learning model
and stores them in a PostgreSQL database. It uses Evidently for metrics calculation
and Prefect for workflow orchestration. The script performs a backfill for a
30-day period.
"""
import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from prefect import task, flow

from evidently import Report
from evidently import DataDefinition
from evidently import Dataset
from evidently.metrics.column_statistics import QuantileValue, ValueDrift

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"  # Configure basic logging
)

# --- Constants and Configuration ---

SEND_TIMEOUT = 10  # Timeout in seconds between metric calculation cycles

# Database connection strings
CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"
rand = random.Random()

# SQL statement to create the metrics table. Drops the table if it already exists.
create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	quantile_fare float,
	pred_drift float
)
"""

# --- Data and Model Loading ---

# Load the reference dataset, which serves as a baseline for drift calculations.
reference_data = pd.read_parquet("data/reference.parquet")

# Load the pre-trained linear regression model.
with open("models/lin_reg.bin", "rb") as f_in:
    model = joblib.load(f_in)

# Load the raw data for the month to be processed.
raw_data = pd.read_parquet("data/green_tripdata_2024-03.parquet")

# --- Evidently Report Configuration ---

begin = datetime.datetime(2024, 3, 1, 0, 0)

# Define numerical and categorical features for the model.
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

# Define the data schema for Evidently, including the prediction column.
data_definition = DataDefinition(
    numerical_columns=num_features + ["prediction"],
    categorical_columns=cat_features,
)

# Create an Evidently report to calculate specific metrics.
report = Report(
    metrics=[
        QuantileValue(column="fare_amount", quantile=0.5),  # Median of fare_amount
        ValueDrift(column="prediction"),  # Drift in the model's predictions
    ],
    include_tests=True,
)


@task
def prep_db():
    """
    Prefect task to set up the database. It creates the 'test' database if it
    doesn't exist and then creates the 'dummy_metrics' table.
    """
    with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(create_table_statement)


@task
def calculate_metrics_postgresql(i):
    """
    Prefect task that calculates metrics for a specific day and stores them in PostgreSQL.
    Args:
        i (int): The day index (offset from the 'begin' date).
    """
    # Filter the raw data to get records for the current day.
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ]

    # Generate predictions for the current day's data.
    # Missing values in features are filled with 0 before prediction.
    current_data["prediction"] = model.predict(
        current_data[num_features + cat_features].fillna(0)
    )

    # Create Evidently Dataset objects for current and reference data.
    current_dataset = Dataset.from_pandas(
        current_data, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(
        reference_data, data_definition=data_definition
    )

    # Run the Evidently report to compare the current data against the reference data.
    run = report.run(reference_data=reference_dataset,
                     current_data=current_dataset)

    # Convert the report run results to a dictionary.
    result = run.dict()

    # Extract the calculated metrics from the result dictionary.
    quantile_fare = result["metrics"][0]["value"]
    pred_drift = result["metrics"][1]["value"]

    # Insert the calculated metrics into the PostgreSQL database.
    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "insert into dummy_metrics(timestamp, quantile_fare, pred_drift) values (%s, %s, %s)",
                (begin + datetime.timedelta(i), quantile_fare, pred_drift), # Timestamp for the day
            )


@flow
def batch_monitoring_backfill():
    """
    The main Prefect flow that orchestrates the backfill process.
    It prepares the database and then iterates through 30 days, calculating
    and storing metrics for each day.
    """
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    # Loop through 30 days to perform the backfill.
    for i in range(0, 30):
        calculate_metrics_postgresql(i)

        # Throttle the execution to simulate a periodic job, waiting if necessary.
        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)
        logging.info("data sent")


if __name__ == "__main__":
    # Entry point to run the Prefect flow when the script is executed directly.
    batch_monitoring_backfill()
