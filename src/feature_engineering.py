import pandas as pd
import numpy as np

def create_date_features(df: pd.DataFrame, date_col: str = 'Order_Date') -> pd.DataFrame:
    """
    Creates date-based and cyclic seasonal features.
    """
    df = df.copy()
    
    # Ensure datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Date Features
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Quarter'] = df[date_col].dt.quarter
    # use ISO week number to avoid warnings in newer pandas versions
    df['Week'] = df[date_col].dt.isocalendar().week.astype(int)
    df['Day_of_Week'] = df[date_col].dt.dayofweek # 0 = Monday, 6 = Sunday
    df['Is_Weekend'] = df['Day_of_Week'].map(lambda x: 1 if x >= 5 else 0)
    
    # 2. Cyclic Features for Monthly seasonality
    # Month Sin/Cos captures the circularity of months (Jan and Dec are close)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df

def build_time_series_grid(df: pd.DataFrame, date_col: str = 'Order_Date') -> pd.DataFrame:
    """
    Builds a complete daily grid for each Product_Code and Region combination.
    This ensures that lag and rolling statistics are calculated over consecutive calendar days.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Find overall min and max dates
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Generate complete date range
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Find all unique combinations of Product_Code, Region and their corresponding Alloy_Type / Industry
    groups = df[['Product_Code', 'Region', 'Alloy_Type', 'Industry']].drop_duplicates()
    
    # Create multi-index of all dates and groups
    mux = pd.MultiIndex.from_product(
        [all_dates, groups['Product_Code'].unique(), groups['Region'].unique()],
        names=[date_col, 'Product_Code', 'Region']
    )
    
    grid = pd.DataFrame(index=mux).reset_index()
    
    # Merge back group attributes (Alloy_Type and Industry)
    grid = pd.merge(grid, groups, on=['Product_Code', 'Region'], how='left')
    
    # Daily aggregation of quantity and revenue (sum in case there are multiple transactions on same day)
    daily_agg = df.groupby([date_col, 'Product_Code', 'Region']).agg({
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    # Merge aggregated sales into our grid
    grid = pd.merge(grid, daily_agg, on=[date_col, 'Product_Code', 'Region'], how='left')
    
    # Fill sales gaps with 0
    grid['Quantity'] = grid['Quantity'].fillna(0)
    grid['Revenue'] = grid['Revenue'].fillna(0)
    
    # Sort
    grid = grid.sort_values([ 'Product_Code', 'Region', date_col]).reset_index(drop=True)
    return grid

def create_lag_and_rolling_features(df: pd.DataFrame, date_col: str = 'Order_Date') -> pd.DataFrame:
    """
    Creates lag and rolling statistics features.
    Assumes df is sorted by Product_Code, Region, and Order_Date.
    """
    df = df.copy()
    
    # Group by Product_Code and Region
    grouped = df.groupby(['Product_Code', 'Region'])
    
    # 1. Lag Features (1, 7, 30 days)
    df['Lag_1'] = grouped['Quantity'].shift(1)
    df['Lag_7'] = grouped['Quantity'].shift(7)
    df['Lag_30'] = grouped['Quantity'].shift(30)
    
    # 2. Rolling Statistics
    # 7-day rolling mean
    df['Rolling_Mean_7'] = grouped['Quantity'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    # 30-day rolling mean
    df['Rolling_Mean_30'] = grouped['Quantity'].transform(
        lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
    )
    # 7-day rolling std
    df['Rolling_Std_7'] = grouped['Quantity'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    ).fillna(0)
    
    return df

def generate_all_features(df: pd.DataFrame, date_col: str = 'Order_Date') -> pd.DataFrame:
    """
    Applies the full feature engineering pipeline.
    """
    # Build complete daily grid to align timeseries correctly
    grid = build_time_series_grid(df, date_col)
    
    # Create date-based and cyclic features
    grid = create_date_features(grid, date_col)
    
    # Create lag and rolling features
    grid = create_lag_and_rolling_features(grid, date_col)
    
    # Drop rows where we cannot compute lag features (due to NaN values at the start of series)
    # Lag_30 is the longest lag, so we drop the first 30 rows per group or drop NaN values in Lag_30
    grid = grid.dropna(subset=['Lag_30']).reset_index(drop=True)
    
    return grid

if __name__ == "__main__":
    import os
    # Local verification if run directly
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv"))
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        from preprocessing import clean_data
        df_cleaned = clean_data(df_raw)
        df_features = generate_all_features(df_cleaned)
        print(f"Feature engineering verified. Engineered shape: {df_features.shape}")
        print("Engineered columns:", list(df_features.columns))
    else:
        print(f"Data file not found at {data_path}. Run data_loader.py first.")
