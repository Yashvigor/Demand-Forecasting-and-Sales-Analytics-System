import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the alloy sales dataset by:
    1. Converting Order_Date to datetime
    2. Removing duplicates
    3. Handling missing values
    4. Sorting date-wise
    """
    print(f"Initial shape: {df.shape}")
    
    # 1. Convert Order_Date to datetime
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    
    # Check for invalid dates and drop them
    initial_invalid = df['Order_Date'].isna().sum()
    if initial_invalid > 0:
        print(f"Dropping {initial_invalid} records with invalid/incorrect dates.")
        df = df.dropna(subset=['Order_Date'])
        
    # 2. Remove duplicate rows
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - df.shape[0]
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate records.")
        
    # 3. Handle missing values
    # Fill missing Quantity and Revenue with 0 or impute based on business logic
    missing_qty = df['Quantity'].isna().sum()
    if missing_qty > 0:
        print(f"Handling {missing_qty} missing values in 'Quantity'. Filling with median.")
        df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
        
    missing_rev = df['Revenue'].isna().sum()
    if missing_rev > 0:
        print(f"Handling {missing_rev} missing values in 'Revenue'. Recalculating based on quantity and median price.")
        # We can impute the revenue using Quantity * average price of that Alloy Type
        df['Revenue'] = df['Revenue'].fillna(df['Quantity'] * (df['Revenue'] / df['Quantity']).median())
        
    # Standardize string fields (remove whitespace, correct casing)
    for col in ['Alloy_Type', 'Region', 'Industry', 'Product_Code', 'Customer_ID']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Remove negative or zero quantity sales (as order transactions should be positive)
    invalid_sales = df[df['Quantity'] <= 0].shape[0]
    if invalid_sales > 0:
        print(f"Dropping {invalid_sales} records with negative or zero Quantity.")
        df = df[df['Quantity'] > 0]
        
    # 4. Sort date-wise
    df = df.sort_values(by='Order_Date').reset_index(drop=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    import os
    # Local verification if run directly
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv"))
    if os.path.exists(data_path):
        df_raw = pd.read_csv(data_path)
        df_cleaned = clean_data(df_raw)
        print("Preprocessing verified successfully.")
    else:
        print(f"Data file not found at {data_path}. Run data_loader.py first.")
