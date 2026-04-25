# ⚡ AI-Based Smart Home Energy Consumption Forecasting & Optimization

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/DeepLearning-TensorFlow%2FKeras-FF6F00?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning pipeline that forecasts household energy consumption, detects anomalies, and optimizes electricity costs using XGBoost, LSTM, and an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Optimization Results](#optimization-results)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Authors](#authors)
- [References](#references)

---

## 🔍 Overview

Traditional energy management relies on static rules and manual monitoring — making it inefficient for modern smart homes. This project addresses that gap with a 5-phase AI pipeline:

1. **Data Ingestion & Preprocessing** — Cleans 2M+ minute-level readings into 34,589 hourly records
2. **Feature Engineering** — Builds 20+ predictors from lag variables, rolling statistics, and calendar features
3. **Model Training & Evaluation** — Trains and benchmarks Random Forest, XGBoost, LSTM v1, and LSTM v2
4. **Optimization Engine** — Detects peaks, flags anomalies, and calculates cost savings via load shifting
5. **Interactive Dashboard** — A 6-page Streamlit app with real-time Plotly visualizations

> 📌 A critical **data leakage issue** (R² inflated to 0.9991 due to `Global_intensity`) was identified and corrected, bringing the honest benchmark to R² = 0.5937.

---

## ✨ Features

- 🔮 **Energy Forecasting** — Hourly consumption prediction using XGBoost and LSTM models
- ⚡ **Peak Hour Detection** — Identifies hours 19:00–21:00 as high-consumption periods
- 🚨 **Anomaly Detection** — Rolling Z-score method flags 293 unusual consumption spikes (0.85% of data)
- 💰 **Cost Optimization** — Simulates 20% load shifting to save ~INR 320/month (5.70% reduction)
- 🏠 **Appliance Scheduler** — Recommends optimal run times for washing machines, ACs, EVs, and more
- 📊 **Interactive Dashboard** — 6-page Streamlit app with KPI cards, date filtering, and Plotly charts

---

## 🖥️ Demo

| Dashboard Page | Preview |
|---|---|
| Overview & KPIs | Avg Power: 1.05 kW · Total Usage: 8,312 kWh · Total Cost: ₹59,205 · Savings: ₹3,106 |
| Energy Forecasting | XGBoost Actual vs Predicted · Feature Importance · Error Distribution |
| Cost Analysis | Monthly Actual vs Optimized · Monthly Savings Chart |
| Anomaly Detection | 65 anomalies flagged · Max Spike: 5.63 kW |
| Appliance Scheduler | Recommended time windows per appliance |
| AI Recommendations | Rule-based insights with estimated monthly savings |

---

## 📁 Project Structure

```
SMART_HOME_ENERGY/
│
├── dashboard/
│   └── app.py                      # Main Streamlit dashboard
│
├── data/                           # Raw & processed datasets
│   ├── household_power_consumption.txt
│   ├── cleaned_hourly.csv
│   ├── baseline_predictions.csv
│   ├── cost_analysis.csv
│   └── anomaly_detection.csv
│
├── models/                         # Trained & serialized models
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── lstm_model.keras
│   ├── scaler_multi.pkl
│   ├── feature_list.pkl
│   └── n_features.pkl
│
├── notebooks/                      # Jupyter development notebooks
│   ├── phase1_eda.ipynb
│   ├── phase2_features_model.ipynb
│   ├── phase3_lstm.ipynb
│   └── phase4_optimization.ipynb
│
├── src/                            # Core source modules
├── venv/                           # Virtual environment
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**UCI Individual Household Electric Power Consumption**  
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/235)  
- Records: 2,075,259 minute-level readings (2006–2010)  
- After preprocessing: 34,589 clean hourly records  
- Target variable: `Global_active_power` (kW)

> ⚠️ Download the dataset manually and place `household_power_consumption.txt` inside the `data/` directory before running the notebooks.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-home-energy.git
cd smart-home-energy
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) and place the file at:
```
data/household_power_consumption.txt
```

---

## 🧪 Usage

### Run the notebooks in order

```bash
jupyter notebook
```

| Notebook | Purpose |
|---|---|
| `phase1_eda.ipynb` | Exploratory data analysis & visualization |
| `phase2_features_model.ipynb` | Feature engineering, Random Forest & XGBoost training |
| `phase3_lstm.ipynb` | LSTM v1 (univariate) and LSTM v2 (multivariate) |
| `phase4_optimization.ipynb` | Peak detection, anomaly detection, cost optimization |

### Launch the Streamlit dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📊 Model Performance

All models were trained on 27,536 hours (80%) and evaluated on 6,885 hours (20%), covering Feb–Nov 2010.

| Model | MAE (kW) | RMSE (kW) | R² | MAPE (%) |
|---|---|---|---|---|
| Random Forest | 0.3236 | 0.4723 | 0.5821 | 43.69% |
| **XGBoost ★** | **0.3199** | **0.4657** | **0.5937** | **42.71%** |
| LSTM v1 (Univariate) | 0.3354 | 0.4954 | 0.5404 | 43.58% |
| LSTM v2 (Multivariate) | 0.3238 | 0.4789 | 0.5704 | 39.84% ✓ |

★ Best overall model · ✓ Best MAPE

**Key finding:** `lag_1h` (previous hour's consumption) accounts for **70.7%** of XGBoost feature importance, confirming the strong auto-correlation in residential energy time series.

> 📌 An R² of ~0.60 is the practical ceiling for this dataset without occupancy or weather data. The remaining 40% of variance is driven by unpredictable human behavior.

---

## 💡 Optimization Results

| Module | Result |
|---|---|
| Peak Hours Identified | 19:00, 20:00, 21:00 (1.9× above average) |
| Anomalies Detected | 293 readings (0.85% of dataset) |
| Total Electricity Cost (4 years) | INR 2,69,854.80 |
| Optimized Cost (20% load shift) | INR 2,54,467.21 |
| **Total Savings** | **INR 15,387.59 (~INR 320/month)** |
| Savings Percentage | 5.70% |

Tariff structure used: INR 6.50/kWh (normal) · INR 9.75/kWh (peak)

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas 3.x, NumPy 2.x |
| Machine Learning | Scikit-learn, XGBoost 2.x |
| Deep Learning | TensorFlow 2.x / Keras |
| Dashboard | Streamlit 1.56.0 |
| Interactive Charts | Plotly 6.7.0 |
| Static Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Statistical Analysis | SciPy |
| Development | VS Code + Jupyter |

---

## 🔭 Future Work

- 🌦️ **Weather API Integration** — Open-Meteo API for temperature/humidity data (target R² > 0.75)
- 🏠 **Occupancy Sensing** — PIR sensors or smartphone location (target R² > 0.85)
- ☁️ **Cloud Deployment** — Streamlit Community Cloud or AWS EC2
- 📡 **Real-Time IoT** — Live smart meter data via MQTT protocol
- 🤖 **Transformer Models** — Temporal Fusion Transformer for long-range dependencies
- 📱 **Alert System** — Automated anomaly alerts via Twilio/SendGrid

---

## 👨‍💻 Authors

**Department of Artificial Intelligence and Data Science**  
Rizvi College of Engineering, University of Mumbai  
Mumbai, Maharashtra, India

- Ansari Gulam Rabbani  
- Asif Ansari  
- Huzaifa Bhati  
- Arman Shaikh  

**Guide:** Prof. Ram Maurya  
**Head of Department:** Prof. Junaid Mandviwala

📄 [Published on IJERT](https://ems.ijert.org/submission-successful)

---

## 📚 References

1. Kong, W., Dong, Z. Y., and Hill, D. J., *Short-Term Residential Load Forecasting Based on Gradient Boosting Regression Trees*, IEEE Press, 2019.
2. Shi, H., Xu, M., and Li, R., "A Comparative Study of Machine Learning Methods for Short-Term Household Energy Consumption Prediction," *IEEE Transactions on Smart Grid*, Vol. 9, No. 4, 2018.
3. Marino, D. L., Amarasinghe, K., and Manic, M., "Building Energy Load Forecasting Using Deep Neural Networks," *IEEE IECON*, 2016.
4. Guo, Z. et al., "Deep Learning Based Household Load Forecasting," *IEEE ICEI*, 2018.
5. Himeur, Y. et al., "Artificial Intelligence Based Anomaly Detection of Energy Consumption in Buildings," *Applied Energy*, Vol. 287, 2021.
6. Hochreiter, S., and Schmidhuber, J., "Long Short-Term Memory," *Neural Computation*, Vol. 9, No. 8, 1997.
7. Chen, T., and Guestrin, C., "XGBoost: A Scalable Tree Boosting System," *ACM SIGKDD*, 2016.
8. Hebrail, G., and Berard, A., "UCI Household Electric Power Consumption Dataset," UCI ML Repository, 2012.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
