📈 Stock Market Anomaly Detector
Capstone Project | Quantitative Finance & Data Science
Developed by: Burhanuddin Udaipurwala
![alt text](https://static.streamlit.io/badges/streamlit_badge_svg?style=flat)

![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
📝 Project Overview
The Stock Market Anomaly Detector is an interactive dashboard designed to identify "Black Swan" events and unusual trading behavior in real-time financial data. By utilizing statistical Z-scores and market breadth indicators, the tool distinguishes between isolated stock volatility and systemic market stress.
This project was built to help analysts visualize historical crashes (like the March 2020 COVID-19 panic) and detect price/volume spikes that deviate from historical norms.
✨ Key Features
Statistical Detection Engine: Uses rolling Z-scores to identify price shocks and volume surges.
Market Breadth Analysis: Tracks the percentage of stocks moving in unison to identify systemic risk.
Interactive Date Query: "Point-in-time" analysis tool to investigate specific historical dates.
Dynamic Visualization: Multi-panel Plotly charts showing Price, Indicators, and Market Stress.
Exportable Reports: Download detected anomalies as CSV for further quantitative research.
User-Defined Sensitivity: Adjustable sliders for Return and Volume thresholds.
🛠️ Tech Stack
Language: Python 3.x
Framework: Streamlit (Web Dashboard)
Data Source: yfinance (Yahoo Finance API)
Analysis: Pandas, NumPy, Scikit-Learn
Visualization: Plotly (Interactive Charts)
🔬 How the "Engine" Works
The detector identifies three specific types of anomalies:
Price Anomalies (Return Z-Score):
Calculated by comparing the daily return to its 63-day rolling mean and standard deviation.
Z
=
x
−
μ
σ
Z= 
σ
x−μ
​
 

Flagged if 
∣
Z
∣
>
Threshold
∣Z∣>Threshold
.
Volume Shocks:
Identifies unusual trading activity by analyzing Log-Volume deviations over a 21-day window.
Volatility Range:
Analyzes the High-Low spread relative to the closing price, flagged at the 95th percentile of historical volatility.
Systemic Stress (Market Breadth):
If a high percentage of the selected portfolio is flagged simultaneously and "Market Breadth" (stocks advancing) falls below 30%, a Market Anomaly is triggered.
🚀 Installation & Local Execution
Clone the Repository:
code
Bash
git clone https://github.com/your-username/stock-anomaly-detector.git
cd stock-anomaly-detector
Create a Virtual Environment (Optional but Recommended):
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
code
Bash
pip install -r requirements.txt
Run the Dashboard:
code
Bash
streamlit run streamlit_app.py
📂 Project Structure
code
Text
├── streamlit_app.py      # Main application code
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── research/
    └── colab_notebook.ipynb # Initial EDA and model research
📊 Example Use Case: March 2020 COVID Crash
To see the tool in action:
Select stocks: AAPL, MSFT, NVDA, QQQ.
Set Date Range: 2018-01-01 to 2020-04-01.
Run Analysis.
Use the Date Query Tool for 2020-03-12.
Result: The dashboard will trigger a 🚨 MARKET ANOMALY alert, showing a Breadth of 0% and extreme negative Z-scores across all tickers.
🤝 Contributing
Contributions are welcome! If you have ideas for adding machine learning models (like Isolation Forests) or sentiment analysis integration, feel free to fork the repo and submit a PR.
