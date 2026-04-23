# 📈 Stock Market Anomaly Detector 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?style=flat)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📝 Project Overview
The **Stock Market Anomaly Detector** is an interactive financial dashboard designed to identify "Black Swan" events and unusual trading behavior in historical and real-time stock data. By utilizing statistical Z-scores and market breadth indicators, the tool distinguishes between isolated stock volatility and systemic market stress.

This project demonstrates the application of **Unsupervised Learning** concepts and **Statistical Process Control** to financial time-series data.

---

## ✨ Key Features
- **Statistical Detection Engine:** Uses rolling Z-scores to identify price shocks and volume surges.
- **Market Breadth Analysis:** Tracks the percentage of stocks moving in unison to identify systemic risk vs. idiosyncratic risk.
- **Interactive Date Query:** A point-in-time analysis tool to investigate specific historical dates (e.g., the COVID-19 crash).
- **Multi-Panel Visualization:** Interactive Plotly charts showing Price, Indicators, and Market Stress in one view.
- **Persistence Layer:** Built using Streamlit `session_state` to ensure data persists across user interactions without re-fetching.
- **Exportable Reports:** Download detected anomalies as a formatted CSV for further quantitative research.

---

## 🛠️ Tech Stack
- **Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Acquisition:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API)
- **Data Processing:** Pandas, NumPy
- **Mathematics/ML:** Scikit-Learn (Standardization & Rolling Statistics)
- **Visualization:** Plotly (Interactive Web Graphics)

---

## 🔬 The Detection Logic
The detector identifies anomalies through three primary mathematical lenses:

1. **Price Anomalies (Return Z-Score):**
   Calculated by comparing the daily return to its 63-day rolling mean ($\mu$) and standard deviation ($\sigma$).
   $$Z = \frac{x - \mu}{\sigma}$$
   *Flagged if $|Z| > \text{User Threshold}$.*

2. **Volume Shocks:**
   Identifies unusual trading activity by analyzing Log-Volume deviations over a 21-day window. Significant volume spikes often precede or confirm price anomalies.

3. **Volatility (Range %):**
   Analyzes the High-Low spread relative to the closing price. Data points in the top 5th percentile of historical volatility are flagged.

4. **Market Breadth:**
   Measures the "health" of the move. If a high percentage of stocks are flagged simultaneously and the percentage of advancing stocks falls below 30%, a **Market Anomaly** is triggered.

---

## 🚀 Installation & Local Execution

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/stock-anomaly-detector.git
   cd stock-anomaly-detector
