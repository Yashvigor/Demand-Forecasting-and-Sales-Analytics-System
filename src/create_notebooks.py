import os
import json

def create_eda_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1: Exploratory Data Analysis (EDA)\n",
                    "## Industrial Demand Forecasting and Sales Analytics System\n",
                    "\n",
                    "This notebook performs the Exploratory Data Analysis for the alloy sales dataset to analyze historical trends, identify seasonality, address missing values/outliers, and answer key business questions:\n",
                    "1. Which alloy sells the most?\n",
                    "2. Which months have the highest demand?\n",
                    "3. Is demand seasonal?\n",
                    "4. Which regions contribute most sales?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import plotly.express as px\n",
                    "import plotly.graph_objects as go\n",
                    "import os\n",
                    "\n",
                    "# Set visualization style\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                    "plt.rcParams['font.size'] = 12"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Load Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = os.path.join(\"..\", \"data\", \"sales_data.csv\")\n",
                    "df = pd.read_csv(data_path)\n",
                    "df['Order_Date'] = pd.to_datetime(df['Order_Date'])\n",
                    "print(f\"Dataset contains {len(df)} records.\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. General Data Audit & Cleaning\n",
                    "We check for missing values, duplicate records, outliers, and our date range."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"--- Missing Values Analysis ---\")\n",
                    "print(df.isnull().sum())\n",
                    "\n",
                    "print(\"\\n--- Duplicate Rows ---\")\n",
                    "print(f\"Number of duplicates: {df.duplicated().sum()}\")\n",
                    "\n",
                    "print(\"\\n--- Date Range ---\")\n",
                    "print(f\"Start Date: {df['Order_Date'].min()}\")\n",
                    "print(f\"End Date: {df['Order_Date'].max()}\")\n",
                    "\n",
                    "print(\"\\n--- Quantity and Revenue Outliers (Description) ---\")\n",
                    "df[['Quantity', 'Revenue']].describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Business Questions Analysis\n",
                    "\n",
                    "#### Q1: Which alloy sells the most?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "alloy_sales = df.groupby('Alloy_Type').agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()\n",
                    "alloy_sales = alloy_sales.sort_values(by='Quantity', ascending=False)\n",
                    "\n",
                    "print(\"--- Alloy Sales Summary ---\")\n",
                    "print(alloy_sales.to_string(index=False))\n",
                    "\n",
                    "# Plot Alloy-wise Sales\n",
                    "fig = px.bar(alloy_sales, x='Alloy_Type', y='Quantity', title='Total Demand (Quantity) by Alloy Type',\n",
                    "             color='Quantity', color_continuous_scale='Viridis')\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Q2 & Q3: Which months have highest demand? Is demand seasonal?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract month and aggregate\n",
                    "df['Month_Num'] = df['Order_Date'].dt.month\n",
                    "df['Month_Name'] = df['Order_Date'].dt.strftime('%B')\n",
                    "\n",
                    "monthly_sales = df.groupby(['Month_Num', 'Month_Name']).agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()\n",
                    "\n",
                    "fig = px.line(monthly_sales, x='Month_Name', y='Quantity', markers=True,\n",
                    "              title='Monthly Demand Seasonality Pattern',\n",
                    "              labels={'Quantity': 'Quantity Sold', 'Month_Name': 'Month'})\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "#### Q4: Which regions contribute most sales?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "region_sales = df.groupby('Region').agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()\n",
                    "region_sales = region_sales.sort_values(by='Revenue', ascending=False)\n",
                    "\n",
                    "print(\"--- Regional Sales Summary ---\")\n",
                    "print(region_sales.to_string(index=False))\n",
                    "\n",
                    "# Plot Region Performance\n",
                    "fig = px.pie(region_sales, values='Revenue', names='Region', title='Sales Revenue Distribution by Region',\n",
                    "             hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Over Time Demand & Revenue Trends\n",
                    "We plot the monthly aggregated demand over the entire timeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_monthly_trend = df.groupby(df['Order_Date'].dt.to_period('M')).agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()\n",
                    "df_monthly_trend['Order_Date'] = df_monthly_trend['Order_Date'].dt.to_timestamp()\n",
                    "\n",
                    "# Plotting\n",
                    "fig = go.Figure()\n",
                    "fig.add_trace(go.Scatter(x=df_monthly_trend['Order_Date'], y=df_monthly_trend['Quantity'],\n",
                    "                    mode='lines+markers', name='Quantity Sold', line=dict(color='royalblue', width=3)))\n",
                    "\n",
                    "fig.update_layout(title='Monthly Demand Over Time (Quantity Trend)',\n",
                    "                  xaxis_title='Date', yaxis_title='Total Quantity')\n",
                    "fig.show()\n",
                    "\n",
                    "fig_rev = go.Figure()\n",
                    "fig_rev.add_trace(go.Scatter(x=df_monthly_trend['Order_Date'], y=df_monthly_trend['Revenue'],\n",
                    "                    mode='lines+markers', name='Revenue ($)', line=dict(color='forestgreen', width=3)))\n",
                    "fig_rev.update_layout(title='Monthly Sales Revenue Over Time',\n",
                    "                  xaxis_title='Date', yaxis_title='Revenue ($)')\n",
                    "fig_rev.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def create_prophet_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 4: Time-Series Forecasting with Prophet\n",
                    "## Industrial Demand Forecasting and Sales Analytics System\n",
                    "\n",
                    "This notebook demonstrates fitting the **Facebook Prophet** model on the alloy sales data. Prophet is selected due to:\n",
                    "- Robustness to daily missing data / irregular gaps.\n",
                    "- Ability to handle B2B weekly purchase cycles and quarterly/annual seasonality.\n",
                    "- Provision of explicit confidence intervals."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from prophet import Prophet\n",
                    "import matplotlib.pyplot as plt\n",
                    "import os\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Load Data\n",
                    "We read in `data/sales_data.csv` and group it by product and region."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = os.path.join(\"..\", \"data\", \"sales_data.csv\")\n",
                    "df = pd.read_csv(data_path)\n",
                    "df['Order_Date'] = pd.to_datetime(df['Order_Date'])\n",
                    "\n",
                    "# Aggregate daily sales per product-region series\n",
                    "ts_data = df.groupby(['Order_Date', 'Product_Code', 'Region'])['Quantity'].sum().reset_index()\n",
                    "print(f\"Time series dataset contains {ts_data.shape[0]} daily records.\")\n",
                    "ts_data.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Fit Prophet on a Single Product-Region Combination\n",
                    "Let's select a single series to visualize its fit and future 90-day forecast."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "prod = ts_data['Product_Code'].unique()[0]\n",
                    "region = ts_data['Region'].unique()[0]\n",
                    "print(f\"Selected series: Product {prod} in Region {region}\")\n",
                    "\n",
                    "single_series = ts_data[(ts_data['Product_Code'] == prod) & (ts_data['Region'] == region)].copy()\n",
                    "\n",
                    "# Fit Prophet required format\n",
                    "prophet_df = single_series[['Order_Date', 'Quantity']].rename(columns={'Order_Date': 'ds', 'Quantity': 'y'})\n",
                    "\n",
                    "# Split train/test (last 90 days as test)\n",
                    "split_date = prophet_df['ds'].max() - pd.Timedelta(days=90)\n",
                    "train = prophet_df[prophet_df['ds'] <= split_date]\n",
                    "test = prophet_df[prophet_df['ds'] > split_date]\n",
                    "\n",
                    "# Fit model\n",
                    "m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)\n",
                    "m.fit(train)\n",
                    "\n",
                    "# Forecast\n",
                    "future = m.make_future_dataframe(periods=90, freq='D')\n",
                    "forecast = m.predict(future)\n",
                    "\n",
                    "# Plot forecast\n",
                    "fig1 = m.plot(forecast)\n",
                    "plt.title(f\"Prophet Forecast for {prod} ({region})\")\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Display Seasonality Components\n",
                    "Prophet allows us to decompose the trend, weekly cycles, and yearly seasonality patterns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig2 = m.plot_components(forecast)\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def create_xgboost_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 4: Time-Series Forecasting with XGBoost\n",
                    "## Industrial Demand Forecasting and Sales Analytics System\n",
                    "\n",
                    "This notebook implements a supervised machine learning approach to forecast alloy demand using **XGBoost**. \n",
                    "\n",
                    "Steps:\n",
                    "1. Load the preprocessed and feature engineered data.\n",
                    "2. Encode categorical columns (`Product_Code`, `Region`, `Alloy_Type`).\n",
                    "3. Chronologically split into train and test sets.\n",
                    "4. Train a global XGBoost Regressor.\n",
                    "5. Predict on test set and evaluate."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import xgboost as xgb\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import os\n",
                    "\n",
                    "sns.set_theme(style=\"whitegrid\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Load Feature Engineered Data\n",
                    "We read in `data/output/engineered_features.csv` generated by the pipeline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "features_path = os.path.join(\"..\", \"data\", \"output\", \"engineered_features.csv\")\n",
                    "if not os.path.exists(features_path):\n",
                    "    print(\"Features file not found. Let's build them from raw data using src/run_pipeline.py first.\")\n",
                    "else:\n",
                    "    df = pd.read_csv(features_path)\n",
                    "    df['Order_Date'] = pd.to_datetime(df['Order_Date'])\n",
                    "    print(f\"Loaded dataset with {df.shape[0]} records and {df.shape[1]} columns.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Supervised Learning Format & Split"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Feature columns\n",
                    "features = [\n",
                    "    'Year', 'Month', 'Quarter', 'Week', 'Day_of_Week', 'Is_Weekend',\n",
                    "    'Month_Sin', 'Month_Cos', 'Lag_1', 'Lag_7', 'Lag_30',\n",
                    "    'Rolling_Mean_7', 'Rolling_Mean_30', 'Rolling_Std_7'\n",
                    "]\n",
                    "\n",
                    "# One-hot encode categoricals\n",
                    "df_encoded = pd.get_dummies(df, columns=['Product_Code', 'Region', 'Alloy_Type'], drop_first=False)\n",
                    "\n",
                    "# Get final feature list\n",
                    "dummy_cols = [c for c in df_encoded.columns if '_' in c and not c.startswith('Rolling_') and not c.startswith('Lag_') and not c.startswith('Month_') and not c.startswith('Day_') and not c.startswith('Is_') and not c.startswith('Order_Date') and not c == 'Quantity' and not c == 'Revenue' and not c == 'Customer_ID' and not c == 'Industry']\n",
                    "all_features = features + dummy_cols\n",
                    "\n",
                    "# Split chronologically (last 90 days as test)\n",
                    "max_date = df_encoded['Order_Date'].max()\n",
                    "split_date = max_date - pd.Timedelta(days=90)\n",
                    "\n",
                    "train_df = df_encoded[df_encoded['Order_Date'] <= split_date].dropna()\n",
                    "test_df = df_encoded[df_encoded['Order_Date'] > split_date].dropna()\n",
                    "\n",
                    "X_train, y_train = train_df[all_features], train_df['Quantity']\n",
                    "X_test, y_test = test_df[all_features], test_df['Quantity']\n",
                    "\n",
                    "print(f\"Training set shape: {X_train.shape}\")\n",
                    "print(f\"Test set shape: {X_test.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Fit XGBoost Model"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "model = xgb.XGBRegressor(\n",
                    "    n_estimators=150,\n",
                    "    learning_rate=0.08,\n",
                    "    max_depth=6,\n",
                    "    random_state=42\n",
                    ")\n",
                    "\n",
                    "model.fit(\n",
                    "    X_train, y_train,\n",
                    "    eval_set=[(X_test, y_test)],\n",
                    "    verbose=False\n",
                    ")\n",
                    "\n",
                    "print(\"Model fitting complete.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Feature Importance\n",
                    "Let's see which features are most important in determining alloy demand."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "importance = pd.DataFrame({\n",
                    "    'Feature': all_features,\n",
                    "    'Importance': model.feature_importances_\n",
                    "}).sort_values(by='Importance', ascending=False).head(10)\n",
                    "\n",
                    "sns.barplot(data=importance, x='Importance', y='Feature', palette='crest')\n",
                    "plt.title('Top 10 Feature Importances - XGBoost')\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return notebook

def save_notebooks(notebooks_dir):
    os.makedirs(notebooks_dir, exist_ok=True)
    
    eda = create_eda_notebook()
    with open(os.path.join(notebooks_dir, "EDA.ipynb"), "w") as f:
        json.dump(eda, f, indent=1)
        
    prophet_nb = create_prophet_notebook()
    with open(os.path.join(notebooks_dir, "Prophet_Model.ipynb"), "w") as f:
        json.dump(prophet_nb, f, indent=1)
        
    xgb_nb = create_xgboost_notebook()
    with open(os.path.join(notebooks_dir, "XGBoost_Model.ipynb"), "w") as f:
        json.dump(xgb_nb, f, indent=1)
        
    print(f"Generated notebook templates successfully in {notebooks_dir}")

if __name__ == "__main__":
    notebooks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
    save_notebooks(notebooks_dir)
