#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# In[2]:

BASE_DIR = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}")

# aktifkan autolog
#mlflow.set_experiment("Energy_Consumption")
mlflow.autolog()


# In[3]:

DATA_DIR = BASE_DIR/"Data"/"data_preprocessed.csv"

df = pd.read_csv(DATA_DIR)
df.head()


# In[4]:


'''
BASE_DIR = Path.cwd()
df = pd.read_csv("energy_consumption_preprocessing.csv")
#df.head()
'''


# In[5]:


# Pisahkan fitur dan target
X = df.drop(columns=["Energy Consumption"])
y = df["Energy Consumption"]


# In[6]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[7]:


# Training model dan tracking ke MLflow
with mlflow.start_run():

    model = RandomForestRegressor(
    random_state=42

    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 : {r2:.4f}")


# In[8]:

#print("Tracking URI:", mlflow.get_tracking_uri())

#!pipreqs . --force

