#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:58:18 2025

@author: pdpai24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


df = pd.read_csv("ai_job_dataset.csv")

df['required_skills'] = df['required_skills'].dropna()

features = ['required_skills', 'education_required', 'job_title', 'years_experience']
target = 'salary_usd'
X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(max_features=1000), 'required_skills'),
    ('edu_ohe', OneHotEncoder(handle_unknown='ignore'), ['education_required']),
    ('job_ohe', OneHotEncoder(handle_unknown='ignore'), ['job_title']),
    ('exp_scaler', StandardScaler(), ['years_experience'])
])

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

input_dim = X_train_transformed.shape[1]

def build_model():
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_model()
history = model.fit(X_train_transformed.toarray(), y_train, 
                    validation_split=0.1,
                    epochs=20, batch_size=32, verbose=1)

y_pred = model.predict(X_test_transformed.toarray()).flatten()

print("\nSample predictions:")
print(pd.DataFrame({'Actual': y_test.values[:5], 'Predicted': y_pred[:5]}))

print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



y_pred = model.predict(X_test_transformed.toarray()).flatten()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

y_pred_scaled = model.predict(X_test_transformed.toarray())
y_pred = scaler.inverse_transform(y_pred_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.tight_layout()
plt.show()