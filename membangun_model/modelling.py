import mlflow
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
import random 
import numpy as np

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('modelling')

data = pd.read_csv('../membangun_model/data_preprocessing.csv')

X = data.drop(columns=['Churn'], axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



input_example = X_train[0:5]

with mlflow.start_run():

    mlflow.autolog()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path='model',
        input_example=input_example
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", float(accuracy))