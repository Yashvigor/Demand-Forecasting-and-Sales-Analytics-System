import os
import zipfile
import requests
import pandas as pd
import numpy as np
import hashlib

def generate_synthetic_data(filepath, n_records=25000):
    print("Generating high-fidelity synthetic alloy demand dataset...")
    np.random.seed(42)
    
    # Define categories
    alloys = {
        'Alloy A (Aluminum)': {'base_price': 15.0, 'industries': ['Aerospace', 'Automotive', 'Electronics']},
        'Alloy B (Steel)': {'base_price': 8.5, 'industries': ['Construction', 'Automotive', 'Marine']},
        'Alloy C (Copper)': {'base_price': 22.0, 'industries': ['Electronics', 'Construction']},
        'Alloy D (Titanium)': {'base_price': 45.0, 'industries': ['Aerospace', 'Marine']},
        'Alloy E (Nickel)': {'base_price': 30.0, 'industries': ['Aerospace', 'Electronics', 'Chemical']}
    }
    
    regions = ['Western', 'Southern', 'Midwestern', 'Northeastern']
    
    # Generate list of dates (3 years: 2023-01-01 to 2025-12-31)
    dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
    
    # Products list
    products = [f"PROD-{i:03d}" for i in range(1, 21)]
    prod_alloy_map = {prod: list(alloys.keys())[i % len(alloys)] for i, prod in enumerate(products)}
    
    # Create customer pool
    customers = [f"CUST-{i:04d}" for i in range(1001, 1081)]
    
    records = []
    
    # Loop over dates and items to generate sales
    # To keep it realistic and under control, we'll sample active series per day
    for dt in dates:
        # B2B purchasing: higher on weekdays, lower on weekends
        day_of_week = dt.dayofweek
        weekday_factor = 1.2 if day_of_week < 5 else 0.3
        
        # Monthly seasonality (spikes near financial year-end and Q4)
        month = dt.month
        if month in [10, 11, 12]:
            month_factor = 1.3  # Q4 Peak
        elif month in [3, 4]:
            month_factor = 1.25 # Spring rise
        else:
            month_factor = 0.95
            
        # Yearly growth trend
        year_factor = 1.0 + (dt.year - 2023) * 0.08
        
        # Generate sales for a subset of products and regions per day to represent transactions
        n_active_sales = np.random.randint(15, 35)
        for _ in range(n_active_sales):
            prod = np.random.choice(products)
            alloy = prod_alloy_map[prod]
            region = np.random.choice(regions)
            
            # Select customer
            cust = np.random.choice(customers)
            
            # Industry consistent with alloy
            allowed_industries = alloys[alloy]['industries']
            industry = np.random.choice(allowed_industries)
            
            # Calculate base quantity with seasonality and trend
            base_qty = np.random.exponential(scale=150) + 50
            # Apply factors
            qty = int(base_qty * weekday_factor * month_factor * year_factor)
            if qty <= 0:
                qty = np.random.randint(5, 20)
                
            # Sell Price with slight random variation
            base_price = alloys[alloy]['base_price']
            price = round(base_price * np.random.uniform(0.95, 1.05), 2)
            revenue = round(qty * price, 2)
            
            records.append({
                'Order_Date': dt.strftime('%Y-%m-%d'),
                'Alloy_Type': alloy,
                'Quantity': qty,
                'Revenue': revenue,
                'Customer_ID': cust,
                'Region': region,
                'Industry': industry,
                'Product_Code': prod
            })
            
    df = pd.DataFrame(records)
    
    # If we got fewer than 5000 records, force at least 5000
    if len(df) < 5000:
        return generate_synthetic_data(filepath, n_records=5000)
    
    # Sort by date
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df = df.sort_values('Order_Date').reset_index(drop=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}. Shape: {df.shape}")
    return df

def download_and_map_m5(filepath):
    # Zip url for M5 dataset
    m5_url = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
    raw_dir = "data/m5_raw"
    zip_path = os.path.join(raw_dir, "m5.zip")
    
    try:
        os.makedirs(raw_dir, exist_ok=True)
        print("Attempting to download M5 dataset from Nixtla GitHub mirror...")
        
        # Download file with a 15 second timeout to avoid hanging
        response = requests.get(m5_url, timeout=15, stream=True)
        if response.status_code != 200:
            raise Exception(f"Download failed with status code {response.status_code}")
            
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Download complete. Extracting zip archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
            
        print("M5 files extracted successfully. Loading datasets...")
        
        calendar = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))
        sales = pd.read_csv(os.path.join(raw_dir, "sales_train_validation.csv"))
        prices = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
        
        print(f"Loaded calendar ({calendar.shape}), sales ({sales.shape}), prices ({prices.shape}).")
        
        # Select subset of products to keep it performant
        # We sample 20 items to represent different Alloy product codes
        sampled_items = sales['item_id'].drop_duplicates().sample(20, random_state=42).tolist()
        sales_sub = sales[sales['item_id'].isin(sampled_items)].copy()
        
        print(f"Sampled {len(sampled_items)} items for training. Melting sales dataset...")
        
        # Melt sales from wide (d_1, d_2...) to long format
        day_cols = [c for c in sales_sub.columns if c.startswith('d_')]
        # Keep only the last 2 years of daily data (approx d_1183 to d_1913) to speed up training
        day_cols_sub = day_cols[-730:]
        
        melted = pd.melt(
            sales_sub,
            id_vars=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            value_vars=day_cols_sub,
            var_name='d',
            value_name='Quantity'
        )
        
        print(f"Melted dataset shape: {melted.shape}")
        
        # Clean calendar
        calendar_sub = calendar[['date', 'd', 'wm_yr_wk']].copy()
        
        # Merge sales with calendar
        df_merged = pd.merge(melted, calendar_sub, on='d', how='inner')
        
        # Merge sales with prices
        df_merged = pd.merge(df_merged, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='inner')
        
        print(f"Merged dataset shape: {df_merged.shape}")
        
        # Filter out 0 quantities to make it realistic for order dates
        df_merged = df_merged[df_merged['Quantity'] > 0].reset_index(drop=True)
        
        # Mapping to Alloys schema
        alloy_map = {
            'HOBBIES': 'Alloy A (Aluminum)',
            'HOUSEHOLD': 'Alloy B (Steel)',
            'FOODS': 'Alloy C (Copper)'
        }
        
        region_map = {
            'CA': 'Western',
            'TX': 'Southern',
            'WI': 'Midwestern'
        }
        
        industries = ['Aerospace', 'Automotive', 'Construction', 'Electronics', 'Marine']
        
        # Create final mappings
        df_merged['Order_Date'] = pd.to_datetime(df_merged['date'])
        df_merged['Alloy_Type'] = df_merged['cat_id'].map(lambda x: alloy_map.get(x, 'Alloy D (Titanium)'))
        df_merged['Region'] = df_merged['state_id'].map(lambda x: region_map.get(x, 'Northeastern'))
        df_merged['Product_Code'] = df_merged['item_id']
        df_merged['Revenue'] = round(df_merged['Quantity'] * df_merged['sell_price'], 2)
        
        # Deterministic Customer ID based on product and store
        def make_cust_id(row):
            h = hashlib.md5(f"{row['item_id']}_{row['store_id']}".encode()).hexdigest()
            return f"CUST-{int(h[:4], 16) % 1000 + 1000}"
            
        df_merged['Customer_ID'] = df_merged.apply(make_cust_id, axis=1)
        
        # Deterministic Industry based on store_id hash
        def make_industry(row):
            idx = int(hashlib.md5(row['store_id'].encode()).hexdigest()[:4], 16) % len(industries)
            return industries[idx]
            
        df_merged['Industry'] = df_merged.apply(make_industry, axis=1)
        
        # Select final columns
        final_cols = ['Order_Date', 'Alloy_Type', 'Quantity', 'Revenue', 'Customer_ID', 'Region', 'Industry', 'Product_Code']
        df_final = df_merged[final_cols].copy()
        
        # Sort by date
        df_final = df_final.sort_values('Order_Date').reset_index(drop=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_final.to_csv(filepath, index=False)
        print(f"Dataset successfully mapped and saved to {filepath}. Shape: {df_final.shape}")
        
        # Clean up large temporary zip to save space
        try:
            os.remove(zip_path)
            print("Cleaned up M5 zip file to free space.")
        except Exception:
            pass
            
        return df_final
        
    except Exception as e:
        print(f"Error occurred during M5 dataset processing: {e}")
        # Fall back to high-fidelity synthetic generator
        return generate_synthetic_data(filepath)

if __name__ == "__main__":
    sales_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv"))
    download_and_map_m5(sales_data_path)
