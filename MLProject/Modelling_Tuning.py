#!/usr/bin/env python
# coding: utf-8

# In[1]:


# modelling_tuning.py

import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# In[4]:


import mlflow

BASE_DIR = Path(__file__).resolve().parent



# In[4]:


DATA_DIR = BASE_DIR/"Data"/"data_preprocessed.csv"

df = pd.read_csv(DATA_DIR)
df.head()


# In[5]:


'''
BASE_DIR = Path.cwd()
df = pd.read_csv("energy_consumption_preprocessing.csv")
#df.head()
'''


# In[6]:


X = df.drop(columns=["Energy Consumption"])
y = df["Energy Consumption"]


# In[7]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# In[8]:


base_model = RandomForestRegressor(
    random_state=42
)


# In[9]:


# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)


def plot_actual_vs_pred(y_test, y_pred, path="actual_vs_predicted.png"):
    y_true = np.asarray(y_test)
    y_hat = np.asarray(y_pred)
    lo = min(y_true.min(), y_hat.min())
    hi = max(y_true.max(), y_hat.max())

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_hat, alpha=0.5, edgecolor="none")
    plt.plot([lo, hi], [lo, hi], "r--", label="Prediksi sempurna")
    plt.xlabel("Nilai aktual (Energy Consumption)")
    plt.ylabel("Nilai prediksi")
    plt.title("Aktual vs Prediksi")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# In[10]:


with mlflow.start_run():

    rf = RandomForestRegressor(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # manual logging parameter terbaik
    mlflow.log_params(grid_search.best_params_)

    # manual logging metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # simpan model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )

    print("Best Params:", grid_search.best_params_)
    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)

    # artifacts tambahan
    mlflow.log_artifact("feature_importance.csv",artifact_path="CSV")
    mlflow.log_artifact("prediction_results.csv",artifact_path="CSV")
    plot_path = plot_actual_vs_pred(y_test, y_pred)
    mlflow.log_artifact(plot_path, artifact_path="plots")


# In[11]:


#!pipreqs . --force

