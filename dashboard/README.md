# Power BI Setup Guidelines - Industrial Demand Forecasting & Sales Analytics

This document serves as the guide for business stakeholders to reconstruct the **Sales Analytics** and **Demand Forecasting** dashboards in **Power BI Desktop** using the pipeline's output predictions.

## Data Source Setup

1. Open **Power BI Desktop**.
2. Click on **Get Data** > **Text/CSV**.
3. Navigate to the project directory and select:
   `data/output/forecast_predictions.csv`
4. Click **Load**.

---

## Dashboard 1: Sales Analytics Dashboard

This page displays historical sales performance and B2B client transaction patterns.

### 1. KPI Cards (Key Performance Indicators)
Use the **Card** visual for the following metrics. Create the DAX measures below to ensure correct aggregations:

* **Total Sales Revenue**
  ```dax
  Total Revenue = SUM(forecast_predictions[Revenue])
  ```
  *Format as Currency ($), rounded to 2 decimal places.*

* **Total Quantity Sold**
  ```dax
  Total Quantity (kg) = SUM(forecast_predictions[Quantity])
  ```
  *Format as Whole Number with thousands separators.*

* **Total Purchase Orders**
  ```dax
  Total Orders = COUNTROWS(forecast_predictions)
  ```

* **Top Selling Alloy**
  ```dax
  Top Alloy = 
  CALCULATE(
      SELECTEDVALUE(forecast_predictions[Alloy_Type]),
      TOPN(1, VALUES(forecast_predictions[Alloy_Type]), [Total Quantity (kg)], DESC)
  )
  ```

### 2. Charts & Visuals

* **Monthly Sales Trend (Line Chart)**
  * **Axis (X-Axis):** `Order_Date` (Group by Month)
  * **Values (Y-Axis):** `Total Quantity (kg)` or `Total Revenue`
  * *Tip: Enable markers for daily/monthly granular visibility.*

* **Alloy Distribution (Donut Chart)**
  * **Legend:** `Alloy_Type`
  * **Values:** `Quantity` (Sum)

* **Regional Performance (Horizontal Bar Chart)**
  * **Y-Axis:** `Region`
  * **X-Axis:** `Total Revenue` (Sum)
  * *Tip: Add color saturation scale to bars based on revenue.*

* **Industry Demand Breakdown (Stacked Column Chart)**
  * **X-Axis:** `Industry`
  * **Y-Axis:** `Total Quantity (kg)`

---

## Dashboard 2: Demand Forecasting Dashboard

This page displays the historical actuals compared with the Prophet and XGBoost forecast models, plus out-of-sample future projections.

### 1. Actual vs Forecast Trend (Line Chart)

Create a line chart visual to display actual quantities side-by-side with both machine learning model outputs:

* **X-Axis:** `Order_Date`
* **Y-Axis (Multiple Lines):**
  * `Quantity` (Historical Actuals)
  * `Prophet_Forecast` (Prophet Model Output)
  * `XGBoost_Forecast` (XGBoost Model Output)
* **Trend Formatting:**
  * Style the **Actuals** line as solid dark grey/black.
  * Style the **Prophet** line as blue with a dashed pattern.
  * Style the **XGBoost** line as orange with a dotted pattern.

### 2. Forecast Confidence Interval (Line and Clustered Column Chart or Custom Ribbon)

* Add `Prophet_Lower` and `Prophet_Upper` to the tooltips or plot them as thin, semi-transparent lines to indicate the **95% Confidence Interval bounds**. 
* If using the custom **AppSource error bars** visual, set `Prophet_Forecast` as the measure, `Prophet_Lower` as the lower bound, and `Prophet_Upper` as the upper bound.

---

## Interactive Filters (Slicers)

Place these slicers at the top or left pane of the report canvas and sync them across both dashboard pages:

1. **Alloy Category Slicer:** Drag the `Alloy_Type` column.
2. **Sales Region Slicer:** Drag the `Region` column.
3. **Timeline Slicer:** Drag the `Order_Date` column (set as Slider or relative date filter).
4. **Target Industry Slicer:** Drag the `Industry` column.
