# Capstone-4A
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
- Erica Zhao  
- Jianjun Gao  
- Sandhya

---

## ▶️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run notebooks
jupyter notebook

# Step 3: Launch dashboard
streamlit run xxxx.py
