# Capstone-4A
# 🧠 Mental Health Demand & Supply Dashboard

🌐 **[🔗 View Live Dashboard](https://mental-health-dashboard-iw4i.onrender.com)**  
📊 Visualizing mismatches in mental health demand and provider supply across U.S. states using Google Trends and AHRF data. Built with Dash + Plotly.

---

⚠️ **Access Note:**  
If you see a **"Not Private"** warning or **blocked page** on your campus network (e.g., GWU), this is due to security systems flagging newly created domains.  
Please either:

- ✅ Open the link using a **personal network / mobile hotspot / VPN**
- ✅ Or click "visit this website anyway" if prompted by your browser

This Render-hosted dashboard is completely safe and part of an academic project.

---
# 🧠 Mental Health Demand and Supply Analysis

## 📌 Problem Statement

### Mismatch in Mental Health Resources  
Despite increasing demand for mental health services, many U.S. regions remain underserved.  

### Market Gap  
Online platforms often lack **geographically tailored strategies** for deploying resources and marketing.

### Potential Impact  
By identifying **high-demand, low-supply regions**, we aim to support:
- Mental health providers
- Policymakers
- Telehealth platforms

---

## 🎯 Objectives & Solutions

- Map state-level mismatch between **demand (search data)** and **supply (provider density)**
- Forecast mental health demand trends
- Build interactive tools to support decision-making

---

## 🧾 Data Sources

### 📈 Google Trends (Demand)  
- Weekly search volume of keywords like “depression treatment,” per U.S. state  
- Collected via `pytrends` API  
- Time range: 5 years (1 year processed)

### 🏥 AHRF Dataset (Supply)  
- From HRSA’s Area Health Resources Files  
- Includes provider counts, facility distribution  
- [HRSA Website](https://data.hrsa.gov/)

---
## 📁 Data Description

## us_trends_monthly_cleaned.csv
- Source: Google Trends
- Content: Monthly search volume for mental health–related terms
- Timeframe: 2019–2024
- Scope: U.S. state-level

## combined_depression_anxiety.csv
- Source: AHRF + survey data
- Content: Merged depression and anxiety indicators

---

## 📂 Notebooks

Explore our data analysis and modeling work in the following notebooks:

- 📊 [EDA: Google Trends (US Monthly)](notebooks/eda_linear_regression_trends.ipynb)
- 📊 [EDA: Combined Depression & Anxiety](notebooks/eda_linear_regression_combined.ipynb)
- 🤖 [XGBoost: Anxiety Model](notebooks/xgboost_model_anxiety.ipynb)
- 🤖 [XGBoost: Depression Model](notebooks/xgboost_model_depression.ipynb)
- 🔍 [XGBoost: SHAP Interpretation](notebooks/xgboost_combined_shap.ipynb)

## 🧪 Methodology

### 📊 Exploratory Data Analysis (EDA)  
- Identify **demand-supply mismatch** zones  

### 🔮 Predictive Modeling  
- **Time Series Forecasting**: LSTM / ARIMA  
- **Regression**: Linear, Ridge, XGBoost  
- **Geospatial**: Map mental health deserts (Plotly / Folium)

---

## 💻 Deliverables

- 📍 **Interactive Dashboard**  
  Visualize trends across states using Streamlit or Plotly Dash

- 📊 **Modeling Notebooks**  
  Organized by theme (anxiety, depression, SHAP, etc.)

- 📂 **Cleaned Data Files**  
  In CSV format for reproducibility

---

## 🛠️ Tech Stack

- Python, Pandas, Numpy, Scikit-learn, XGBoost, Pytrends  
- Jupyter Notebook, Streamlit, Plotly  
- Git, GitHub

---
## 👥 Team Members

- Qibin Huang  
- Jianjun Gao  
- Sandhya Karki
- Erica Zhao

---

## ▶️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run notebooks
jupyter notebook

# Step 3: Launch dashboard (Dash App)
cd App
python combined_app.py
