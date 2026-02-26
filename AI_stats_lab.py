
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


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    data = load_diabetes()
    X, y = data.data, data.target

    # Standardize entire dataset for CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5,
        scoring='r2'
    )

    mean_r2 = scores.mean()
    std_r2 = scores.std()

    """
    Standard deviation represents model stability.
    Lower std → model performs consistently across folds.

    Cross-validation reduces variance risk by:
    - Training/testing on multiple splits
    - Giving more reliable performance estimate
    """

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    """
    False Negative (Medical Meaning):
    A false negative means:
    The model predicts "no cancer"
    BUT the patient actually has cancer.

    This is dangerous because:
    - Disease goes untreated
    - Condition may worsen
    """

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1, 10, 100]
    results = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        results[C] = (train_acc, test_acc)

    """
    When C is very small:
    - Strong regularization
    - Model underfits
    - Lower train accuracy

    When C is very large:
    - Weak regularization
    - Model fits training data very closely
    - Risk of overfitting

    Overfitting usually happens when C is very large.
    """

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5,
        scoring='accuracy'
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    """
    Why CV is critical in medical diagnosis:

    - Medical errors are costly.
    - Single train-test split may be misleading.
    - Cross-validation ensures model performs well
      across multiple data partitions.
    - Reduces risk of deploying unstable model.
    """

    return mean_accuracy, std_accuracy
