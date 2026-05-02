import numpy as np
import pandas as pd

# this file creates the data we train on and cleans it up

def generate_coffee_data(n_days=365, seed=42):
    np.random.seed(seed)
    day_of_week = np.random.randint(0, 7, n_days)
    temperature = np.random.uniform(5, 42, n_days)
    is_raining = np.random.choice([0, 1], n_days, p=[0.7, 0.3])
    has_event = np.random.choice([0, 1], n_days, p=[0.85, 0.15])
    social_buzz = np.random.randint(0, 50, n_days)

    revenue = (
        15000 # base daily revenue (PKR)
        + (day_of_week >= 5) * 6000 # weekends earn more
        + np.clip((25 - temperature), 0, 20) * 200 # cool weather = more customers
        - is_raining * 3000 # rain hurts foot traffic
        + has_event * 8000 # nearby events = big boost
        + social_buzz * 150 # social media matters
        + np.random.randn(n_days) * 1500 # random daily noise
    )

    revenue = np.clip(revenue, 5000, 50000) # keep it in a realistic range

    # put everything in a DataFrame so it's readable
    df = pd.DataFrame({
        "day_of_week":  day_of_week,
        "temperature":  temperature,
        "is_raining":   is_raining,
        "has_event":    has_event,
        "social_buzz":  social_buzz,
        "revenue":      revenue
    })

    return df

def prepare_data(df):
    # split into features (X) and target (y)
    feature_cols = ["day_of_week", "temperature", "is_raining", "has_event", "social_buzz"]
    X = df[feature_cols].values.astype(float)
    y = df["revenue"].values.astype(float)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    # 80% training, 20% testing
    split = int(0.8 * len(df))
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y_norm[:split], y_norm[split:]

    scale_info = {"y_min": y_min, "y_max": y_max}

    return X_train, X_test, y_train, y_test, scale_info
