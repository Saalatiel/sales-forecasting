# 📈 Sales Forecasting with Machine Learning

This project performs **monthly revenue forecasting** using time series analysis and Machine Learning models trained on the **Superstore Dataset**.

---

## 🚀 Project Overview

The script automates the entire data pipeline: loading real-world sales data, performing feature engineering for time series, training three different models, and generating a revenue forecast for the next 6 months.

---

## 📊 Dataset

We use the [Superstore Sales Dataset](https://raw.githubusercontent.com/mikemooreviz/superstore/master/superstore.csv) — public data from an American retail chain containing order records between 2014 and 2017.

---

## 🤖 Models Implemented

| Model | Type |
|---|---|
| **Linear Regression** | Linear baseline |
| **Random Forest** | Tree-based ensemble |
| **Gradient Boosting** | Sequential boosting |

> **Selection Criteria:** The model with the lowest **MAPE** (Mean Absolute Percentage Error) on the test set is selected for the final forecast. In most scenarios, **Random Forest** delivers the highest accuracy.

---

## 🛠️ Feature Engineering

To capture seasonality and trends, the following features are generated:

| Feature | Description |
|---|---|
| `month_n` | Month of the year (1–12) |
| `qtr` | Quarter |
| `year` | Year |
| `l1`, `l2`, `l3` | Sales from the previous 3 months (Lag features) |
| `ma3` | 3-month Moving Average |
| `ma6` | 6-month Moving Average |

---

## 📈 Results & Visualizations

The following charts are automatically generated and saved in the `/img` folder when running the script:

### 1. Historical Time Series
![series](img/serie_temporal.png)

### 2. Seasonality & Trends
![seasonality](img/sazonalidade.png)

### 3. Predicted vs. Actual (Test Set)
![prediction](img/previsao_vs_real.png)

### 4. 6-Month Revenue Forecast
![forecast](img/forecast.png)

### 5. Feature Importance Ranking
![importance](img/feature_importance.png)

---

## 💻 How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/sales-forecasting.git](https://github.com/your-username/sales-forecasting.git)
cd sales-forecasting

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![pandas](https://img.shields.io/badge/pandas-1.3+-green)
