# %%
import os
import sagemaker
import boto3
import pandas as pd

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %% [markdown]
# ### Configure connection with aws

# %%
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# %%
sm = boto3.client("sagemaker")
session = sagemaker.Session()
region = session.boto_session.region_name
bucket = AWS_S3_BUCKET_NAME

# %% [markdown]
# ### Load Local Data

# %%
raw_train_data_filepath = "data/raw/train.csv"
df = pd.read_csv(raw_train_data_filepath)

# %%
# separe into feature and label

df["price_range"].value_counts(normalize=True)

label = "price_range"
features = df.drop(label, axis=1).columns

x = df[features]
y = df[label]

# %%
# verify the data structure
x.head()

# %%
x.columns

# %%
x.describe()

# %%
y.head()

# %%
y.value_counts()

# %%
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

# %%
# join
train_split = pd.concat([X_train, y_train], axis=1)
test_split = pd.concat([X_test, y_test], axis=1)

# %%
local_train_split_filepath = "data/split/train.csv"
local_test_split_filepath = "data/split/test.csv"

# %%
# save the data into a local folder
train_split.to_csv(local_train_split_filepath, index=False)
test_split.to_csv(local_test_split_filepath, index=False)

# %%
# send data splits to s3
data_prefix = "sagemaker/mobile_price_classification/sklearncontainer"

train_split_path = session.upload_data(
    path=local_train_split_filepath, bucket=bucket, key_prefix=data_prefix
)
test_split_path = session.upload_data(
    path=local_test_split_filepath, bucket=bucket, key_prefix=data_prefix
)

# %% [markdown]
# ### Train and Score a model using Sagemaker

# %%
from sagemaker.sklearn.estimator import SKLearn

# load configurations
AWS_SAGEMAKER_ROLE = os.getenv("AWS_SAGEMAKER_ROLE")
AWS_SAGEMAKER_INSTANCE_TYPE = os.getenv("AWS_SAGEMAKER_INSTANCE_TYPE")
SAGEMAKER_FRAMEWORK_VERSION = "1.2-1"

# build the estimator
sklearn_estimator = SKLearn(
    entry_point="script.py",
    role=AWS_SAGEMAKER_ROLE,
    instance_count=1,
    instance_type=AWS_SAGEMAKER_INSTANCE_TYPE,
    framework_version=SAGEMAKER_FRAMEWORK_VERSION,
    base_job_name="RF-sklearn",
    hyperparameters={
        "n_estimators": 100,
        "random_state": 0,
    },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600,
)

# %%
# train a model
sklearn_estimator.fit({"train": train_split_path, "test": test_split_path}, wait=True)

# %% [markdown]
# ### Load back the trained model

# %%
sklearn_estimator.latest_training_job.wait(logs=None)
# search the model artifact
artifact = sm.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)

# %%
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime

model_name = "sklearn-model-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model = SKLearnModel(
    name=model_name,
    model_data=artifact,
    role=AWS_SAGEMAKER_ROLE,
    entry_point="script.py",
    framework_version=SAGEMAKER_FRAMEWORK_VERSION,
)

# %% [markdown]
# ### Deploy an endpoint

# %%
endpoint_name = "sklearn-model-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)

# %%
raw_test_data_filepath = "data/raw/test.csv"

X_prod = pd.read_csv(raw_test_data_filepath)

# %%
# Make predictions for the unseen data
predictions = predictor.predict(X_prod[features].values.tolist())

print(f"predictions: {predictions}")

# %% [markdown]
# ### Delete endpoint

# %%
sm.delete_endpoint(EndpointName=endpoint_name)
