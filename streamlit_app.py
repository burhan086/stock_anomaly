import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings

# Ignore warnings for a cleaner dashboard
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .anomaly-alert { background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336; color: #b71c1c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Stock Market Anomaly Detector</p>', unsafe_allow_html=True)
st.markdown("**Capstone Project** | Student: Burhanuddin Udaipurwala")
st.divider()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📊 Select Stocks")
    available_stocks = ['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
    selected_stocks = st.multiselect(
        "Choose stocks to analyze",
        available_stocks,
        default=['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN']
    )
    
    st.subheader("📅 Date Range")
    start_date = st.date_input("Start Date", value=pd.to_datetime('2018-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2020-04-01'))
    
    st.subheader("🎚️ Detection Thresholds")
    ret_threshold = st.slider("Return Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    vol_threshold = st.slider("Volume Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    range_threshold = st.slider("Range Percentile Threshold", 90, 99, 95, 1)
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("💡 Tip: Lower thresholds = more sensitive detection")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty: continue
        # Handle multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Fix for newer yfinance versions where Adj Close might be missing
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        data[t] = df[['Open','High','Low','Close','Adj Close','Volume']]
    return data

def create_features(df, ticker):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['LogVol'] = np.log(df['Volume'].replace(0, 1))
    
    # Z-scores based on rolling windows
    ret_mean = df['Return'].shift(1).rolling(63).mean()
    ret_std = df['Return'].shift(1).rolling(63).std()
    df['ret_z'] = (df['Return'] - ret_mean) / ret_std
    
    vol_mean = df['LogVol'].shift(1).rolling(21).mean()
    vol_std = df['LogVol'].shift(1).rolling(21).std()
    df['volz'] = (df['LogVol'] - vol_mean) / vol_std
    
    rng = (df['High'] - df['Low']) / df['Close']
    df['range_pct'] = rng.rolling(63).rank(pct=True) * 100
    
    df['Ticker'] = ticker
    df['Price'] = df['Adj Close']
    return df.dropna()

def detect_anomalies(df, ret_thresh, vol_thresh, range_thresh):
    df['anomaly_flag'] = (
        (np.abs(df['ret_z']) > ret_thresh) | 
        (df['volz'] > vol_thresh) | 
        (df['range_pct'] > range_thresh)
    ).astype(int)
    
    def label_type(row):
        if not row['anomaly_flag']: return "Normal"
        types = []
        if np.abs(row['ret_z']) > ret_thresh:
            types.append("crash" if row['Return'] < 0 else "spike")
        if row['volz'] > vol_thresh:
            types.append("volume_shock")
        return "+".join(types) if types else "anomaly"
    
    df['type'] = df.apply(label_type, axis=1)
    return df

def compute_market_metrics(df):
    """Aggregates data to market-level. Fixed for single-stock stability."""
    # Group by Date (index level 0)
    market = df.groupby(level=0).agg({
        'Return': 'mean',
        'anomaly_flag': 'mean'
    }).rename(columns={'Return': 'market_ret', 'anomaly_flag': 'flag_rate'})
    
    # Calculate Breadth (percentage of stocks that were positive today)
    # Using a robust method that works for 1 or many stocks
    market['breadth'] = df['Return'].gt(0).groupby(level=0).mean()
    
    # Market anomaly logic
    # Use 95th percentile if enough data, else default to 5% move
    thresh = market['market_ret'].abs().quantile(0.95) if len(market) > 20 else 0.05
    
    market['market_anomaly_flag'] = (
        (market['market_ret'].abs() > thresh) | (market['breadth'] < 0.3)
    ).astype(int)
    
    return market

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

if run_analysis and len(selected_stocks) > 0:
    with st.spinner("Processing Market Data..."):
        data_map = fetch_data(selected_stocks, start_date, end_date)
        
        if not data_map:
            st.error("No data found for the selected tickers/dates.")
        else:
            all_df = pd.concat([create_features(data_map[t], t) for t in data_map]).sort_index()
            test = all_df.copy()
            test = detect_anomalies(test, ret_threshold, vol_threshold, range_threshold)
            market = compute_market_metrics(test)
            
            st.success("Analysis Complete!")

            # 1. OVERVIEW METRICS
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Trading Days", f"{len(test.index.unique())}")
            with col2: st.metric("Anomalies", f"{test['anomaly_flag'].sum()}", f"{test['anomaly_flag'].mean()*100:.1f}%")
            with col3: st.metric("Market Stress Days", f"{market['market_anomaly_flag'].sum()}")
            with col4: st.metric("Avg Breadth", f"{market['breadth'].mean()*100:.1f}%")

            # 2. DATE QUERY
            st.header("🔍 Date Query Tool")
            query_date = st.date_input("Pick a date", value=test.index.max().date(), 
                                       min_value=test.index.min().date(), max_value=test.index.max().date())
            
            q_str = str(query_date)
            
            # Check if date exists in our analyzed data
            if q_str in market.index.astype(str):
                m = market.loc[q_str]
                
                # Market Status Display
                if m['market_anomaly_flag']:
                    st.markdown(f'<div class="anomaly-alert">🚨 MARKET ANOMALY: Return {m["market_ret"]*100:.2f}%, Breadth {m["breadth"]*100:.0f}%</div>', unsafe_allow_html=True)
                else:
                    st.info(f"✅ Market Status: Normal | Return: {m['market_ret']*100:.2f}% | Breadth: {m['breadth']*100:.0f}%")
                
               
                # We use a boolean mask to ensure we ALWAYS get a DataFrame, even for 1 stock
                day_data = test[test.index.astype(str) == q_str]
                
                # Filter for only anomalous stocks on that day
                day_anoms = day_data[day_data['anomaly_flag'] == 1]
                
                if not day_anoms.empty:
                    st.subheader(f"Flagged Stocks ({len(day_anoms)})")
                    # Displaying specific columns for clarity
                    st.dataframe(day_anoms[['Ticker', 'type', 'Return', 'ret_z', 'volz']].style.format({
                        'Return': '{:.2%}',
                        'ret_z': '{:.2f}',
                        'volz': '{:.2f}'
                    }))
                else:
                    st.success("No individual stock anomalies detected on this date.")
               
            else:
                st.warning("No trading data available for the selected date (it might be a weekend or holiday).")

            # 3. CHARTS
            st.header("📈 Interactive Visualization")
            chart_stock = st.selectbox("Select stock", selected_stocks)
            s_data = test[test['Ticker'] == chart_stock]
            s_anoms = s_data[s_data['anomaly_flag'] == 1]

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
            fig.add_trace(go.Scatter(x=s_data.index, y=s_data['Price'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_anoms.index, y=s_anoms['Price'], mode='markers', name='Anomaly', marker=dict(color='red', size=8)), row=1, col=1)
            fig.add_trace(go.Scatter(x=s_data.index, y=s_data['ret_z'], name='Ret Z-Score'), row=2, col=1)
            fig.add_trace(go.Scatter(x=market.index, y=market['breadth'], name='Market Breadth', fill='tozeroy'), row=3, col=1)
            fig.update_layout(height=800, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # 4. DOWNLOAD DATA
            st.header("📋 Download Report")
            csv = test[test['anomaly_flag']==1].to_csv().encode('utf-8')
            st.download_button("Download Anomalies CSV", csv, "anomalies.csv", "text/csv")

else:
    st.info("👈 Select stocks and click 'Run Analysis' in the sidebar.")

st.caption("Burhanuddin Udaipurwala")
