import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

# Ignore warnings
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
    .anomaly-alert { background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336; color: #b71c1c; margin: 10px 0; }
    .normal-info { background-color: #e8f5e9; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; color: #1b5e20; margin: 10px 0; }
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
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.test_df = None
        st.session_state.market_df = None
        st.rerun()

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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        data[t] = df[['Open','High','Low','Close','Adj Close','Volume']]
    return data

def create_features(df, ticker):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['LogVol'] = np.log(df['Volume'].replace(0, 1))
    
    # Rolling Z-scores (Standardizing relative to history)
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
    """Aggregates data to market-level. Robust for single-stock stability."""
    market = df.groupby(level=0).agg({
        'Return': 'mean',
        'anomaly_flag': 'mean'
    }).rename(columns={'Return': 'market_ret', 'anomaly_flag': 'flag_rate'})
    
    # Breadth: Percentage of stocks moving up
    market['breadth'] = df['Return'].gt(0).groupby(level=0).mean()
    
    # Threshold for market stress
    thresh = market['market_ret'].abs().quantile(0.95) if len(market) > 20 else 0.05
    market['market_anomaly_flag'] = (
        (market['market_ret'].abs() > thresh) | (market['breadth'] < 0.3)
    ).astype(int)
    
    return market

# ============================================================================
# SESSION STATE MANAGEMENT (Prevents Page Resets)
# ============================================================================
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'market_df' not in st.session_state:
    st.session_state.market_df = None

# ============================================================================
# EXECUTION LOGIC
# ============================================================================

# When button is clicked
if run_analysis and len(selected_stocks) > 0:
    with st.spinner("🚀 Analyzing Market Data..."):
        raw_data = fetch_data(selected_stocks, start_date, end_date)
        
        if not raw_data:
            st.error("❌ No data found. Please check your stock symbols or dates.")
        else:
            # Feature Engineering
            processed_list = [create_features(raw_data[t], t) for t in raw_data]
            all_df = pd.concat(processed_list).sort_index()
            
            # Detection
            test_results = detect_anomalies(all_df, ret_threshold, vol_threshold, range_threshold)
            market_results = compute_market_metrics(test_results)
            
            # Store in session state
            st.session_state.test_df = test_results
            st.session_state.market_df = market_results
            st.success("✅ Analysis Complete!")

# DISPLAY DASHBOARD IF DATA EXISTS
if st.session_state.test_df is not None:
    test = st.session_state.test_df
    market = st.session_state.market_df

    # 1. OVERVIEW METRICS
    st.header("📊 Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Trading Days", f"{len(test.index.unique())}")
    with col2: st.metric("Total Anomalies", f"{test['anomaly_flag'].sum()}", f"{test['anomaly_flag'].mean()*100:.1f}% rate")
    with col3: st.metric("Market Stress Days", f"{market['market_anomaly_flag'].sum()}")
    with col4: st.metric("Avg Market Breadth", f"{market['breadth'].mean()*100:.1f}%")
    st.divider()

    # 2. DATE QUERY TOOL (Robust Fix)
    st.header("🔍 Date Query Tool")
    query_date = st.date_input("Select a date to investigate", 
                               value=test.index.max().date(), 
                               min_value=test.index.min().date(), 
                               max_value=test.index.max().date())
    
    q_str = str(query_date)
    if q_str in market.index.astype(str):
        m = market.loc[q_str]
        
        if m['market_anomaly_flag']:
            st.markdown(f'<div class="anomaly-alert">🚨 <b>MARKET ANOMALY DETECTED</b><br>Market Return: {m["market_ret"]*100:.2f}% | Breadth: {m["breadth"]*100:.0f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="normal-info">✅ <b>MARKET STATUS: NORMAL</b><br>Market Return: {m["market_ret"]*100:.2f}% | Breadth: {m["breadth"]*100:.0f}%</div>', unsafe_allow_html=True)
        
        # Safe dataframe extraction for single or multiple stocks
        day_data = test[test.index.astype(str) == q_str]
        day_anoms = day_data[day_data['anomaly_flag'] == 1]
        
        if not day_anoms.empty:
            st.subheader(f"Flagged Stocks ({len(day_anoms)})")
            st.dataframe(day_anoms[['Ticker', 'type', 'Return', 'ret_z', 'volz']].style.format({
                'Return': '{:.2%}',
                'ret_z': '{:.2f}',
                'volz': '{:.2f}'
            }), use_container_width=True)
        else:
            st.success("No individual stock anomalies detected on this date.")
    else:
        st.warning("⚠️ No trading data available for this date (Market Holiday/Weekend).")
    
    st.divider()

    # 3. INTERACTIVE VISUALIZATION
    st.header("📈 Interactive Visualization")
    chart_stock = st.selectbox("Select a stock to visualize", selected_stocks)
    
    s_data = test[test['Ticker'] == chart_stock]
    s_anoms = s_data[s_data['anomaly_flag'] == 1]

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{chart_stock} Price & Anomalies', 'Z-Score Indicators', 'Market Breadth')
    )

    # Price Panel
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['Price'], name='Price', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_anoms.index, y=s_anoms['Price'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)

    # Indicators Panel
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['ret_z'], name='Return Z', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['volz'], name='Volume Z', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=ret_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-ret_threshold, line_dash="dash", line_color="red", row=2, col=1)

    # Breadth Panel
    fig.add_trace(go.Scatter(x=market.index, y=market['breadth'], name='Breadth', fill='tozeroy', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=3, col=1)

    fig.update_layout(height=800, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # 4. DOWNLOAD DATA
    st.header("📋 Export Results")
    full_anomalies = test[test['anomaly_flag'] == 1].copy()
    csv = full_anomalies.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download All Anomalies as CSV",
        data=csv,
        file_name=f"market_anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    # Initial Welcome Screen
    st.info("👈 Use the sidebar to select your stocks and click 'Run Analysis' to see the results.")
    st.markdown("""
    ### About this Project
    This dashboard identifies **Stock Market Anomalies** using statistical thresholds:
    *   **Price Shocks:** Movements exceeding the chosen Z-Score.
    *   **Volume Shocks:** Log-volume spikes relative to the 21-day average.
    *   **Volatility:** Intraday range percentiles.
    *   **Market Breadth:** Identifying if anomalies are isolated or market-wide.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("📈 Stock Market Anomaly Detector | Burhanuddin Udaipurwala")
