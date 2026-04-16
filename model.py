from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_model(df):
    features = [
        'base_price',
        'discount_pct',
        'current_sell_price',
        'fb_ad_spend',
        'tiktok_ad_spend',
        'affiliate_commission_rate'
    ]
    
    X = df[features]
    y = df['sales_quantity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    importance = model.feature_importances_

    return model, rmse, r2, features, importance
