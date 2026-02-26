
"""
Linear & Logistic Regression Lab
Only numpy and sklearn used.
random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1: Load dataset
    data = load_diabetes()
    X, y = data.data, data.target

    # STEP 2: Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6: Top 3 features (largest absolute coefficients)
    coefficients = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coefficients)[-3:][::-1].tolist()

    """
    Overfitting Analysis:
    If train R² is much higher than test R², model may overfit.
    For LinearRegression on this dataset, overfitting is usually mild.

    Why scaling is important:
    - Features may have different magnitudes.
    - Scaling ensures coefficients are comparable.
    - Prevents dominance of large-scale features.
    """

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices

