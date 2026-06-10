import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def create_report_visualizations(combined_path, output_dir):
    df = pd.read_csv(combined_path)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Alloy Distribution
    plt.figure(figsize=(6, 4))
    alloy_sales = df.groupby('Alloy_Type')['Quantity'].sum().reset_index().sort_values(by='Quantity', ascending=False)
    sns.barplot(data=alloy_sales, x='Quantity', y='Alloy_Type', hue='Alloy_Type', palette='viridis', legend=False)
    plt.title('Total Demand (Quantity) by Alloy Type', fontsize=12, fontweight='bold')
    plt.xlabel('Quantity Sold')
    plt.ylabel('')
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "alloy_demand.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    
    # Chart 2: Monthly Seasonality Trend
    plt.figure(figsize=(6, 4))
    df['Month_Name'] = df['Order_Date'].dt.strftime('%b')
    df['Month_Num'] = df['Order_Date'].dt.month
    monthly_trend = df.groupby(['Month_Num', 'Month_Name'])['Quantity'].sum().reset_index()
    sns.lineplot(data=monthly_trend, x='Month_Name', y='Quantity', marker='o', color='royalblue', linewidth=2.5)
    plt.title('Monthly Demand Seasonal Pattern', fontsize=12, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Quantity Sold')
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "monthly_seasonality.png")
    plt.savefig(chart2_path, dpi=150)
    plt.close()

    # Chart 3: Model Forecast Comparison (Last 90 days)
    plt.figure(figsize=(6, 4))
    max_date = df['Order_Date'].max()
    test_days = 90
    split_date = max_date - pd.Timedelta(days=test_days)
    test_df = df[df['Order_Date'] > split_date].copy()
    
    # Daily aggregation for comparison
    daily_comp = test_df.groupby('Order_Date')[['Quantity', 'Prophet_Forecast', 'XGBoost_Forecast']].sum().reset_index()
    
    plt.plot(daily_comp['Order_Date'], daily_comp['Quantity'], label='Actual', color='black', alpha=0.6, linewidth=1.5)
    plt.plot(daily_comp['Order_Date'], daily_comp['Prophet_Forecast'], label='Prophet Forecast', color='blue', linestyle='--', linewidth=2)
    plt.plot(daily_comp['Order_Date'], daily_comp['XGBoost_Forecast'], label='XGBoost Forecast', color='orange', linestyle=':', linewidth=2)
    
    plt.title('Actual vs Model Forecasts (Test Set)', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Aggregate Daily Units')
    plt.legend(frameon=True)
    plt.xticks(rotation=15)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "forecast_comparison.png")
    plt.savefig(chart3_path, dpi=150)
    plt.close()

    return chart1_path, chart2_path, chart3_path

class AlloyPDFReport(FPDF):
    def header(self):
        # Top branding header
        self.set_fill_color(24, 43, 73) # Deep Blue
        self.rect(0, 0, 210, 35, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 5, 'INDUSTRIAL ALLOY DEMAND FORECASTING', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font('Helvetica', 'I', 11)
        self.cell(0, 10, 'Executive Demand Planning & Sales Analytics Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()} | Confidential B2B Planning Report', border=0, align='C')

def build_pdf_report(combined_path, comparison_path, insights_text_path, pdf_report_path):
    # Setup visualization output directory
    vis_dir = os.path.dirname(pdf_report_path)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Generate PNGs
    chart1, chart2, chart3 = create_report_visualizations(combined_path, vis_dir)
    
    # Load comparison table and insights
    comp_df = pd.read_csv(comparison_path)
    
    with open(insights_text_path, 'r') as f:
        insights_lines = f.readlines()
        
    pdf = AlloyPDFReport(orientation='P', unit='mm', format='A4')
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    
    # Section 1: Executive Summary
    pdf.set_text_color(24, 43, 73)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '1. Executive Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_fill_color(24, 43, 73)
    pdf.rect(15, pdf.get_y(), 180, 0.5, 'F') # Horizontal rule
    pdf.ln(3)
    
    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', '', 10)
    summary_text = (
        "This report outlines the demand behavior and predictive modeling outcomes for our industrial alloy categories "
        "developed utilizing structural profiles from the M5 Forecasting framework. Overstocking and stockout hazards "
        "represent critical cost drivers in manufacturing operations. By analyzing daily transactional trends across "
        "multiple sales regions and vertical industries, this forecasting system offers predictive visibility and "
        "decision support for procurement, production planning, and sales managers."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)
    
    # Section 2: Core Business Insights (Two Column layout concept)
    pdf.set_text_color(24, 43, 73)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '2. Actionable Business Insights', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.rect(15, pdf.get_y(), 180, 0.5, 'F')
    pdf.ln(3)
    
    pdf.set_text_color(50, 50, 50)
    pdf.set_font('Helvetica', '', 10)
    for line in insights_lines[3:]: # Skip header
        if line.strip():
            # Check for bullet point
            pdf.set_font('Helvetica', 'B', 10)
            pdf.write(5, line[:3]) # Number prefix
            pdf.set_font('Helvetica', '', 10)
            pdf.multi_cell(0, 5, line[3:])
            pdf.ln(2)
            
    # Section 3: Visualizations Page
    pdf.add_page()
    
    pdf.set_text_color(24, 43, 73)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '3. Sales Distribution & Seasonality Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.rect(15, pdf.get_y(), 180, 0.5, 'F')
    pdf.ln(5)
    
    # Place images side-by-side or stacked
    # Stacked with explanations
    pdf.image(chart1, x=15, w=85)
    pdf.image(chart2, x=105, y=pdf.get_y() - 56, w=85) # Offset to make side-by-side
    
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Figure 1: Demand volumes categorized by Alloy Types (Left) and aggregated monthly seasonality (Right) showing typical peak trends in Q4.", align='C')
    pdf.ln(6)
    
    # Section 4: Model Evaluation
    pdf.set_text_color(24, 43, 73)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, '4. Model Evaluation & Comparison', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.rect(15, pdf.get_y(), 180, 0.5, 'F')
    pdf.ln(4)
    
    # Comparison table in PDF
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(24, 43, 73)
    
    # Header cells
    pdf.cell(45, 7, 'Metric', border=1, align='C', fill=True)
    pdf.cell(45, 7, 'Prophet Model', border=1, align='C', fill=True)
    pdf.cell(45, 7, 'XGBoost Model', border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C', fill=True)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    for index, row in comp_df.iterrows():
        # Clean cell inputs
        metric = str(row['Metric'])
        p_val = f"{row['Prophet']:.2f}"
        x_val = f"{row['XGBoost']:.2f}"
        
        pdf.cell(45, 7, metric, border=1, align='C')
        pdf.cell(45, 7, p_val, border=1, align='C')
        pdf.cell(45, 7, x_val, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        
    pdf.ln(6)
    
    # Place Forecast Comparison Chart
    pdf.image(chart3, x=45, w=120)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Figure 2: Chronological evaluation comparison of actual demand versus model outputs over the 90-day out-of-sample test window.", align='C')
    
    pdf.output(pdf_report_path)
    print(f"PDF report successfully compiled at {pdf_report_path}")
    
    # Clean up temporary PNG images
    for img in [chart1, chart2, chart3]:
        try:
            os.remove(img)
        except Exception:
            pass

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    combined = os.path.join(base_dir, "data", "output", "forecast_predictions.csv")
    comparison = os.path.join(base_dir, "data", "output", "model_comparison.csv")
    insights = os.path.join(base_dir, "reports", "insights.txt")
    pdf_report = os.path.join(base_dir, "reports", "insights.pdf")
    
    if os.path.exists(combined) and os.path.exists(comparison) and os.path.exists(insights):
        build_pdf_report(combined, comparison, insights, pdf_report)
    else:
        print("Required input files missing. Run run_pipeline.py first.")
