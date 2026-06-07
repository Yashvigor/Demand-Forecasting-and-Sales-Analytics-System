import pandas as pd
import numpy as np

def calculate_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))

def calculate_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))

def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    # Handle zero division: ignore entries where actual is 0
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)

def calculate_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    sum_actual = np.sum(actual)
    if sum_actual == 0:
        return 0.0
    return float(np.sum(np.abs(actual - forecast)) / sum_actual * 100)

def evaluate_forecasts(df: pd.DataFrame, actual_col: str = 'Quantity', test_days: int = 90) -> pd.DataFrame:
    """
    Computes MAE, RMSE, and MAPE for both Prophet and XGBoost over the last test_days.
    Assumes df has Prophet_Forecast and XGBoost_Forecast columns.
    """
    df = df.copy()
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    
    # Filter to test period
    max_date = df['Order_Date'].max()
    split_date = max_date - pd.Timedelta(days=test_days)
    test_df = df[df['Order_Date'] > split_date].copy()
    
    # Drop rows that don't have predictions from both models
    test_df = test_df.dropna(subset=[actual_col, 'Prophet_Forecast', 'XGBoost_Forecast'])
    
    if len(test_df) == 0:
        print("Warning: Test set is empty. Check dates and forecasting columns.")
        return pd.DataFrame()
        
    actuals = test_df[actual_col].values
    prophet_fcs = test_df['Prophet_Forecast'].values
    xgboost_fcs = test_df['XGBoost_Forecast'].values
    
    # Compute metrics
    metrics = {
        'Metric': ['MAE', 'RMSE', 'MAPE (%)', 'WAPE (%)'],
        'Prophet': [
            calculate_mae(actuals, prophet_fcs),
            calculate_rmse(actuals, prophet_fcs),
            calculate_mape(actuals, prophet_fcs),
            calculate_wape(actuals, prophet_fcs)
        ],
        'XGBoost': [
            calculate_mae(actuals, xgboost_fcs),
            calculate_rmse(actuals, xgboost_fcs),
            calculate_mape(actuals, xgboost_fcs),
            calculate_wape(actuals, xgboost_fcs)
        ]
    }
    
    comparison_df = pd.DataFrame(metrics)
    return comparison_df

def print_comparison_table(comparison_df: pd.DataFrame):
    if comparison_df.empty:
        print("No comparison data available.")
        return
    print("\n" + "="*45)
    print("        MODEL PERFORMANCE COMPARISON")
    print("="*45)
    print(comparison_df.to_string(index=False))
    print("="*45)
    
    # Identify best model based on MAE
    prophet_mae = comparison_df.loc[comparison_df['Metric'] == 'MAE', 'Prophet'].values[0]
    xgb_mae = comparison_df.loc[comparison_df['Metric'] == 'MAE', 'XGBoost'].values[0]
    
    best_model = "Prophet" if prophet_mae < xgb_mae else "XGBoost"
    diff_percent = abs(prophet_mae - xgb_mae) / max(prophet_mae, xgb_mae) * 100
    print(f"Insight: {best_model} performs best with {diff_percent:.1f}% lower MAE.")
    print("="*45)
