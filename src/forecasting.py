import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def run_prophet_forecast(df: pd.DataFrame, date_col: str = 'Order_Date', target_col: str = 'Quantity', forecast_horizon: int = 90) -> tuple[pd.DataFrame, dict]:
    """
    Fits a Prophet model for each unique Product_Code & Region combination.
    Returns:
      - A dataframe containing predictions for historical and future dates.
      - A dictionary of fitted Prophet models per group.
    """
    df = df.copy()
    groups = df.groupby(['Product_Code', 'Region'])
    
    forecast_results = []
    models = {}
    
    print(f"Training Prophet models for {len(groups)} series...")
    
    for (prod, region), group_df in groups:
        # Prepare Prophet dataframe
        prophet_df = group_df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Fit Prophet model
        # Enable weekly and yearly seasonality, disable daily seasonality (since we only have daily records)
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative' # multiplicative seasonality is standard for sales
        )
        model.fit(prophet_df)
        
        # Save model
        models[(prod, region)] = model
        
        # Create future dataframe for the specified horizon
        future = model.make_future_dataframe(periods=forecast_horizon, freq='D')
        forecast = model.predict(future)
        
        # Format output
        fc_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        fc_df = fc_df.rename(columns={'ds': date_col, 'yhat': 'Prophet_Forecast', 'yhat_lower': 'Prophet_Lower', 'yhat_upper': 'Prophet_Upper'})
        fc_df['Product_Code'] = prod
        fc_df['Region'] = region
        fc_df['Alloy_Type'] = group_df['Alloy_Type'].iloc[0]
        
        # Merge with actuals if they exist in the history
        group_actuals = group_df[[date_col, target_col, 'Revenue']].copy()
        fc_df = pd.merge(fc_df, group_actuals, on=date_col, how='left')
        
        forecast_results.append(fc_df)
        
    all_forecasts = pd.concat(forecast_results, ignore_index=True)
    return all_forecasts, models


def prepare_xgboost_data(df: pd.DataFrame, target_col: str = 'Quantity') -> tuple[pd.DataFrame, list, dict]:
    """
    Prepares features for XGBoost, including one-hot encoding for categorical variables.
    """
    df = df.copy()
    
    # Feature lists
    features = [
        'Year', 'Month', 'Quarter', 'Week', 'Day_of_Week', 'Is_Weekend',
        'Month_Sin', 'Month_Cos', 'Lag_1', 'Lag_7', 'Lag_30',
        'Rolling_Mean_7', 'Rolling_Mean_30', 'Rolling_Std_7'
    ]
    
    # Categorical columns to encode
    cat_cols = ['Product_Code', 'Region', 'Alloy_Type']
    
    # We will one-hot encode them to keep the models transparent
    encoded_df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    # Extract list of all engineered columns
    encoded_features = features + [col for col in encoded_df.columns if any(col.startswith(cat + '_') for cat in cat_cols)]
    
    return encoded_df, encoded_features, cat_cols


def run_xgboost_forecast(df: pd.DataFrame, date_col: str = 'Order_Date', target_col: str = 'Quantity', test_days: int = 90) -> tuple[pd.DataFrame, xgb.XGBRegressor, list]:
    """
    Splits the data chronologically, trains a global XGBoost model on the training set,
    and returns predictions for the test set.
    """
    # 1. Prepare data and encode variables
    encoded_df, features, _ = prepare_xgboost_data(df, target_col)
    
    # 2. Chronological Split
    max_date = encoded_df[date_col].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    
    train_mask = encoded_df[date_col] <= split_date
    test_mask = encoded_df[date_col] > split_date
    
    train_df = encoded_df[train_mask].dropna()
    test_df = encoded_df[test_mask].dropna()
    
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    print(f"XGBoost Split Date: {split_date.strftime('%Y-%m-%d')}")
    print(f"XGBoost Train Set: {X_train.shape}, Test Set: {X_test.shape}")
    
    # 3. Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 4. Predict
    # Make a copy of the input dataframe to merge forecasts back
    output_df = df.copy()
    
    # Predict for all matching rows
    all_X = encoded_df[features]
    output_df['XGBoost_Forecast'] = model.predict(all_X)
    
    # Ensure forecast values are non-negative
    output_df['XGBoost_Forecast'] = output_df['XGBoost_Forecast'].clip(lower=0)
    
    return output_df, model, features


def save_trained_models(prophet_models: dict, xgb_model: xgb.XGBRegressor, xgb_features: list, models_dir: str = 'models'):
    """
    Serializes and saves Prophet models, the XGBoost model, and its feature list.
    """
    import os
    import pickle
    import json
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Save Prophet Models (Pickle dictionary)
    prophet_path = os.path.join(models_dir, 'prophet_models.pkl')
    with open(prophet_path, 'wb') as f:
        pickle.dump(prophet_models, f)
    print(f"Saved Prophet models dictionary to {os.path.abspath(prophet_path)}")
    
    # 2. Save XGBoost Model (JSON)
    xgb_path = os.path.join(models_dir, 'xgboost_model.json')
    xgb_model.save_model(xgb_path)
    print(f"Saved XGBoost model to {os.path.abspath(xgb_path)}")
    
    # 3. Save XGBoost Features list (JSON)
    features_path = os.path.join(models_dir, 'xgboost_features.json')
    with open(features_path, 'w') as f:
        json.dump(xgb_features, f)
    print(f"Saved XGBoost features list to {os.path.abspath(features_path)}")

