import os
import numpy as np
import pandas as pd
from data_loader import download_and_map_m5
from preprocessing import clean_data
from feature_engineering import generate_all_features
from forecasting import run_prophet_forecast, run_xgboost_forecast, save_trained_models
from evaluation import evaluate_forecasts, print_comparison_table
from generate_pdf_report import build_pdf_report

def run_pipeline():
    # Filepaths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "sales_data.csv")
    output_dir = os.path.join(base_dir, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("   INDUSTRIAL DEMAND FORECASTING SYSTEM - END-TO-END PIPELINE")
    print("="*60)
    
    # 1. Acquire Data
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Running loader...")
        download_and_map_m5(data_path)
    else:
        print(f"Using existing data file at {data_path}")
        
    df_raw = pd.read_csv(data_path)
    print(f"Loaded raw data with shape: {df_raw.shape}")
    
    # 2. Preprocess Data
    print("\n--- PHASE 2: DATA PREPROCESSING ---")
    df_cleaned = clean_data(df_raw)
    
    # 3. Feature Engineering
    print("\n--- PHASE 3: FEATURE ENGINEERING ---")
    df_features = generate_all_features(df_cleaned)
    print(f"Engineered dataset shape: {df_features.shape}")
    df_features.to_csv(os.path.join(output_dir, "engineered_features.csv"), index=False)
    
    # 4. Forecasting Models
    print("\n--- PHASE 4: RUNNING FORECASTING MODELS ---")
    
    # Run Prophet (horizon = 90 days)
    print("Running Prophet forecasts per product and region (this may take 1-2 minutes)...")
    prophet_forecasts, prophet_models = run_prophet_forecast(df_cleaned, forecast_horizon=90)
    
    # Run XGBoost (test split = 90 days)
    print("Running global XGBoost model...")
    xgboost_results, xgb_model, xgb_features = run_xgboost_forecast(df_features, test_days=90)
    
    # Save models to disk
    models_save_dir = os.path.join(base_dir, "models")
    save_trained_models(prophet_models, xgb_model, xgb_features, models_save_dir)
    
    # 5. Merge Forecasts
    print("\n--- PHASE 5: MERGING FORECAST RESULTS ---")
    # For evaluation and comparisons, we merge both models' forecasts on the historical dataset
    # We rename columns appropriately
    
    # Group Prophet results by Order_Date, Product_Code, Region
    prophet_merge = prophet_forecasts[['Order_Date', 'Product_Code', 'Region', 'Prophet_Forecast', 'Prophet_Lower', 'Prophet_Upper']]
    prophet_merge['Order_Date'] = pd.to_datetime(prophet_merge['Order_Date'])
    
    # XGBoost results already align with df_features Order_Date
    xgboost_merge = xgboost_results[['Order_Date', 'Product_Code', 'Region', 'XGBoost_Forecast']]
    xgboost_merge['Order_Date'] = pd.to_datetime(xgboost_merge['Order_Date'])
    
    # Start with engineered features
    combined = df_features.copy()
    combined['Order_Date'] = pd.to_datetime(combined['Order_Date'])
    
    # Merge Prophet (use outer merge to preserve out-of-sample future forecasts)
    combined = pd.merge(combined, prophet_merge, on=['Order_Date', 'Product_Code', 'Region'], how='outer')
    
    # Post-process merged future rows (interpolate/fill details)
    # Build maps from historical df_features
    lookup = df_features[['Product_Code', 'Region', 'Alloy_Type', 'Industry']].drop_duplicates()
    lookup_alloy = lookup.set_index(['Product_Code', 'Region'])['Alloy_Type'].to_dict()
    lookup_industry = lookup.set_index(['Product_Code', 'Region'])['Industry'].to_dict()
    
    # Find rows representing future forecast (where actual Quantity is NaN)
    future_mask = combined['Quantity'].isna()
    if future_mask.any():
        print(f"Post-processing {future_mask.sum()} future forecast rows...")
        keys = list(zip(combined['Product_Code'], combined['Region']))
        combined['Alloy_Type'] = combined['Alloy_Type'].fillna(pd.Series([lookup_alloy.get(k) for k in keys], index=combined.index))
        combined['Industry'] = combined['Industry'].fillna(pd.Series([lookup_industry.get(k) for k in keys], index=combined.index))
        
        future_dates = combined.loc[future_mask, 'Order_Date']
        combined.loc[future_mask, 'Year'] = future_dates.dt.year
        combined.loc[future_mask, 'Month'] = future_dates.dt.month
        combined.loc[future_mask, 'Quarter'] = future_dates.dt.quarter
        combined.loc[future_mask, 'Week'] = future_dates.dt.isocalendar().week.astype(int)
        combined.loc[future_mask, 'Day_of_Week'] = future_dates.dt.dayofweek
        combined.loc[future_mask, 'Is_Weekend'] = (future_dates.dt.dayofweek >= 5).astype(int)
        
        # Recompute Month Sin/Cos cyclic features
        combined.loc[future_mask, 'Month_Sin'] = np.sin(2 * np.pi * combined.loc[future_mask, 'Month'] / 12)
        combined.loc[future_mask, 'Month_Cos'] = np.cos(2 * np.pi * combined.loc[future_mask, 'Month'] / 12)
    
    # Merge XGBoost
    combined = pd.merge(combined, xgboost_merge, on=['Order_Date', 'Product_Code', 'Region'], how='left')
    
    # Save combined results
    combined_path = os.path.join(output_dir, "forecast_predictions.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined forecast predictions to {combined_path}")
    
    # 6. Evaluation
    print("\n--- PHASE 6: MODEL EVALUATION ---")
    comparison_df = evaluate_forecasts(combined, test_days=90)
    print_comparison_table(comparison_df)
    
    # Save comparison report
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model evaluation comparison to {comparison_path}")
    
    # 7. Generate Actionable Business Insights (Phase 6)
    print("\n--- PHASE 7: BUSINESS INSIGHTS GENERATION ---")
    generate_business_insights(combined, comparison_df, base_dir)
    
    # 8. Compile PDF executive report
    print("\n--- PHASE 8: COMPILING PDF EXECUTIVE REPORT ---")
    pdf_report_path = os.path.join(base_dir, "reports", "insights.pdf")
    insights_text_path = os.path.join(base_dir, "reports", "insights.txt")
    try:
        build_pdf_report(combined_path, comparison_path, insights_text_path, pdf_report_path)
        print("Pipeline PDF Report generated successfully.")
    except Exception as e:
        print(f"Warning: Failed to compile PDF report: {e}")
    
    print("\n" + "="*60)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY")
    print("="*60)

def generate_business_insights(combined: pd.DataFrame, comparison_df: pd.DataFrame, base_dir: str):
    """
    Analyzes historical and forecasted data to print and save major B2B alloy demand insights.
    """
    insights = []
    
    # 1. Top selling alloy
    alloy_sales = combined.groupby('Alloy_Type')['Quantity'].sum().reset_index()
    total_sales = alloy_sales['Quantity'].sum()
    top_alloy = alloy_sales.sort_values(by='Quantity', ascending=False).iloc[0]
    top_alloy_pct = (top_alloy['Quantity'] / total_sales) * 100
    insights.append(f"Top Selling Alloy: {top_alloy['Alloy_Type']} dominates demand, representing {top_alloy_pct:.1f}% of total quantity sold ({int(top_alloy['Quantity']):,} units).")
    
    # 2. Regional Growth
    region_sales = combined.groupby('Region')['Revenue'].sum().reset_index()
    top_region = region_sales.sort_values(by='Revenue', ascending=False).iloc[0]
    insights.append(f"Highest Revenue Region: The {top_region['Region']} region generates the highest sales revenue (${top_region['Revenue']:,.2f}).")
    
    # 3. Monthly / Quarterly Seasonality
    monthly_sales = combined.groupby('Month')['Quantity'].sum()
    q4_sales = combined[combined['Quarter'] == 4]['Quantity'].sum()
    q_avg_sales = combined.groupby('Quarter')['Quantity'].sum().mean()
    q4_increase = (q4_sales - q_avg_sales) / q_avg_sales * 100
    
    if q4_increase > 0:
        insights.append(f"Seasonal Peaks: High demand cycles detected in Q4. Fourth quarter (Q4) order volume is {q4_increase:.1f}% higher than the average quarterly demand.")
    else:
        insights.append("Seasonal Peaks: Monthly demand is steady, but B2B demand shows cyclical buying behavior at the turn of quarters.")
        
    # 4. Weekend vs Weekday B2B Behavior
    weekday_avg = combined[combined['Is_Weekend'] == 0]['Quantity'].mean()
    weekend_avg = combined[combined['Is_Weekend'] == 1]['Quantity'].mean()
    b2b_ratio = (weekday_avg - weekend_avg) / weekend_avg * 100
    insights.append(f"B2B Purchasing Pattern: Daily transaction volume during weekdays is {b2b_ratio:.1f}% higher than weekends, confirming standard industrial corporate purchasing cycles.")
    
    # 5. Best forecasting model
    prophet_mae = comparison_df.loc[comparison_df['Metric'] == 'MAE', 'Prophet'].values[0]
    xgb_mae = comparison_df.loc[comparison_df['Metric'] == 'MAE', 'XGBoost'].values[0]
    best_model = "Prophet" if prophet_mae < xgb_mae else "XGBoost"
    mae_diff = abs(prophet_mae - xgb_mae) / max(prophet_mae, xgb_mae) * 100
    insights.append(f"Forecasting Accuracy: {best_model} model performs best for this dataset, achieving {mae_diff:.1f}% lower Mean Absolute Error than the alternative model.")
    
    # Write to reports/insights.txt (and print)
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    print("\nActionable Business Insights:")
    for insight in insights:
        print(f" - {insight}")
        
    with open(os.path.join(reports_dir, "insights.txt"), "w") as f:
        f.write("INDUSTRIAL ALLOY DEMAND FORECASTING - BUSINESS INSIGHTS\n")
        f.write("========================================================\n\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
            
    print(f"\nSaved text insights report to {os.path.join(reports_dir, 'insights.txt')}")

if __name__ == "__main__":
    run_pipeline()
