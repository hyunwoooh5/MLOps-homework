import os
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


import mlflow
from prefect import task, flow


@task(name="Read DataFrame",
      retries=3,
      retry_delay_seconds=5,
      log_prints=True      
      )
def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    print(f'Number of records: {len(df)}')
    return df


@task(name="Feature Engineering",
      retries=3,
      retry_delay_seconds=5,
      log_prints=True      
      )
def feature_engineering(df):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    
    for col in categorical:
        df[col] = df[col].astype('string')
        
    return df

@task(name="Train Model",
      retries=3,
      retry_delay_seconds=5,
      log_prints=True      
      )
def train_model(df):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"intercept: {lr.intercept_}")
    return lr, dv


@task(name="Log Model",
      retries=3,
      retry_delay_seconds=5,
      log_prints=True      
      )
def log_model(lr, dv):
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("nyc-taxi-hw3")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")

        with open("dict_vectorizer.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dict_vectorizer.pkl")

        mlflow.sklearn.log_model(lr, artifact_path="model")

        run_id = run.info.run_id
        print("Model logged to MLflow")
        print(f"MLflow Run ID: {run_id}")


@flow(name="Main Flow",
      log_prints=True
      )
def main(year, month):
    df = read_dataframe(year, month)
    df = feature_engineering(df)
    lr, dv = train_model(df)
    log_model(lr, dv)


if __name__ == '__main__':
    main(2023, 3)