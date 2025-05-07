from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import boto3
import pathlib
from io import StringIO
import argparse
import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":

    print("INFO: Extracting arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))

    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))

    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    args, _ = parser.parse_known_args()

    print("INFO: Reading data")

    train = pd.read_csv(os.path.join(args.train, args.train_file))
    test = pd.read_csv(os.path.join(args.test, args.test_file))


    print("INFO: Building datasets")

    label = "price_range"
    X_train = train.drop(label, axis=1)
    y_train = train[label]

    X_test = test.drop(label, axis=1)
    y_test = test[label]

    print("INFO: Training RandiForest Model")

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred)

    print("INFO: Model Metrics")

    print("INFO: Accuracy (test): ", acc)
    print("INFO: Classification Report (test): ")
    print(rep)
