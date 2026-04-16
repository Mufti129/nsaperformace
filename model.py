import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_and_compare_models(df):
    # ===== VALIDATION =====
    if df is None or df.empty:
        raise ValueError("DataFrame kosong atau None")

    # ===== FEATURES =====
    features = [
        'base_price',
        'discount_pct',
        'current_sell_price',
        'fb_ad_spend',
        'tiktok_ad_spend',
        'affiliate_commission_rate'
    ]

    for col in features + ['sales_quantity']:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan")

    X = df[features].copy().fillna(0)
    y = df['sales_quantity'].copy().fillna(0)

    # ===== SPLIT =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== MODELS =====
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []

    best_model = None
    best_rmse = float("inf")

    # ===== TRAIN & EVALUATE =====
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "model": name,
            "RMSE": rmse,
            "R2": r2
        })

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values(by="RMSE")

    # ===== FEATURE IMPORTANCE (kalau ada) =====
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
    else:
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": [0]*len(features)
        })

    return best_model, best_model_name, results_df, importance_df

def train_gradient_boosting(df):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    features = [
        'base_price',
        'discount_pct',
        'current_sell_price',
        'fb_ad_spend',
        'tiktok_ad_spend',
        'affiliate_commission_rate'
    ]

    X = df[features].fillna(0)
    y = df['sales_quantity'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2, features

def predict_sales(model, input_data):
    import pandas as pd

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return prediction
