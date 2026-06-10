# Industrial Demand Forecasting and Sales Analytics System

An enterprise-grade forecasting system that predicts future demand for industrial alloys, extracts seasonal demand patterns, compares different modeling approaches, and generates actionable inventory insights.

## Project Overview

In the manufacturing industry, managing inventory is a critical operational challenge. Underestimating demand leads to **stockouts**, resulting in lost revenue and dissatisfied clients. Overestimating demand leads to **overstocking**, which ties up capital and incurs storage costs. 

This system integrates:
1. **Data Preprocessing & Cleaning:** Standardizing timestamps, managing outliers, and imputing missing data.
2. **Feature Engineering:** Crafting date-based, lag, rolling, and cyclic features.
3. **Forecasting Models:**
   - **Prophet:** Captures long-term trends and multiple seasonal patterns.
   - **XGBoost:** Converts time-series into a supervised learning problem to model non-linear interactions.
4. **Model Comparison & Evaluation:** Measuring MAE, RMSE, and MAPE chronologically.
5. **Business Insights & Dashboards:** Translating models into actionable decisions.

---

## Directory Structure

```
Industrial-Demand-Forecasting/
├── data/
│   └── sales_data.csv             # Cleaned & processed alloy sales data
├── models/
│   ├── prophet_models.pkl         # Serialized Prophet models dictionary
│   ├── xgboost_model.json         # Serialized XGBoost regressor model
│   └── xgboost_features.json      # Feature column mapping for XGBoost
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis & Business Questions
│   ├── Prophet_Model.ipynb        # Prophet Training & Forecasting Notebook
│   └── XGBoost_Model.ipynb        # XGBoost Supervised Learning Notebook
├── src/
│   ├── data_loader.py             # Downloads & maps M5 dataset to Alloy domain
│   ├── preprocessing.py           # Preprocessing and cleaning functions
│   ├── feature_engineering.py     # Lags, rolling windows, and cyclical features
│   ├── forecasting.py             # Model training pipelines (Prophet & XGBoost)
│   ├── run_pipeline.py            # End-to-end forecasting pipeline orchestrator
│   └── evaluation.py              # Performance evaluation & comparison metrics
├── app.py                         # Premium Streamlit Dashboard Application
├── requirements.txt               # Project dependency versions
└── README.md                      # System documentation

```

---

## Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.9+** and **Git** installed on your system.

### 2. Clone the Repository
```bash
git clone <repository-url>
cd "Demand Forecasting and Sales Analytics System"
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows Powershell)
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## Technical Workflow

### 1. M5 Dataset Mapping
The project utilizes the structural properties, hierarchy, and real-world patterns of the **M5 Forecasting Dataset** mapped to industrial alloys:
* **Product Categories (Hobbies, Household, Foods)** $\rightarrow$ **Alloy Types (Aluminum, Steel, Copper, Titanium, Nickel)**
* **Store States (CA, TX, WI)** $\rightarrow$ **Regions (Western, Southern, Midwestern)**
* **Item IDs** $\rightarrow$ **Product Codes**
* **Quantities & Sell Prices** $\rightarrow$ **Quantity Sold & Sales Revenue**
* **Customer Segments** $\rightarrow$ **Industry Types (Aerospace, Automotive, Construction, etc.)**

### 2. Run Data Loader
Run the following script to acquire and prepare the dataset:
```bash
python src/data_loader.py
```

### 3. Run Notebooks
Startup Jupyter Notebook to view the steps:
```bash
jupyter notebook
```
Explore:
* `notebooks/EDA.ipynb` for Exploratory Data Analysis.
* `notebooks/Prophet_Model.ipynb` for Prophet modeling.
* `notebooks/XGBoost_Model.ipynb` for XGBoost modeling.

### 4. Run Streamlit Dashboard Web App
To run the interactive B2B Sales and Forecasting dashboard:
```bash
streamlit run app.py
```
This will start the local server and launch the dashboard application in your default web browser (usually at `http://localhost:8501`).

