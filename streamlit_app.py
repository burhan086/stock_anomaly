import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

# Setup and Styling
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Stock Anomaly Detector", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-text { text-align: center; color: #666; margin-bottom: 2rem; }
    .anomaly-alert { background-color: #ffebee; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #f44336; color: #b71c1c; margin: 10px 0; }
    .normal-info { background-color: #e8f5e9; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #4caf50; color: #1b5e20; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Stock Market Anomaly Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text"><b>Capstone Project</b> | Student: Burhanuddin Udaipurwala</p>', unsafe_allow_html=True)
st.divider()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    available_stocks = ['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
    selected_stocks = st.multiselect("Select Stocks", available_stocks, default=['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN'])
    
    start_date = st.date_input("Start Date", value=pd.to_datetime('2018-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2020-04-01'))
    
    st.subheader("🎚️ Sensitivity")
    ret_threshold = st.slider("Return Z-Score (Price)", 1.5, 4.0, 2.5, 0.1, help="How many standard deviations a price must move to be 'weird'. Default is 2.5.")
    vol_threshold = st.slider("Volume Z-Score (Activity)", 1.5, 4.0, 2.5, 0.1, help="Spikes in trading volume relative to 21-day average.")
    range_threshold = st.slider("Range Percentile (Volatility)", 90, 99, 95, 1, help="Flag stocks in the top X% of intraday volatility.")
    
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if st.button("🗑️ Reset Dashboard", use_container_width=True):
        st.session_state.test_df = None
        st.session_state.market_df = None
        st.rerun()

# ============================================================================
# CORE ENGINE
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if 'Adj Close' not in df.columns: df['Adj Close'] = df['Close']
        data[t] = df[['Open','High','Low','Close','Adj Close','Volume']]
    return data

def create_features(df, ticker):
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['LogVol'] = np.log(df['Volume'].replace(0, 1))
    # Rolling stats for Z-Scores
    ret_mean = df['Return'].shift(1).rolling(63).mean()
    ret_std = df['Return'].shift(1).rolling(63).std()
    df['ret_z'] = (df['Return'] - ret_mean) / ret_std
    vol_mean = df['LogVol'].shift(1).rolling(21).mean()
    vol_std = df['LogVol'].shift(1).rolling(21).std()
    df['volz'] = (df['LogVol'] - vol_mean) / vol_std
    df['range_pct'] = ((df['High'] - df['Low']) / df['Close']).rolling(63).rank(pct=True) * 100
    df['Ticker'], df['Price'] = ticker, df['Adj Close']
    return df.dropna()

def detect_anomalies(df, rt, vt, rgt):
    df['anomaly_flag'] = ((np.abs(df['ret_z']) > rt) | (df['volz'] > vt) | (df['range_pct'] > rgt)).astype(int)
    def label_type(row):
        if not row['anomaly_flag']: return "Normal"
        t = []
        if np.abs(row['ret_z']) > rt: t.append("crash" if row['Return'] < 0 else "spike")
        if row['volz'] > vt: t.append("volume_shock")
        return "+".join(t) if t else "volatility_shock"
    df['type'] = df.apply(label_type, axis=1)
    return df

def compute_market_metrics(df):
    m = df.groupby(level=0).agg({'Return': 'mean', 'anomaly_flag': 'mean'}).rename(columns={'Return': 'market_ret', 'anomaly_flag': 'flag_rate'})
    m['breadth'] = df['Return'].gt(0).groupby(level=0).mean()
    thresh = m['market_ret'].abs().quantile(0.95) if len(m) > 20 else 0.05
    m['market_anomaly_flag'] = ((m['market_ret'].abs() > thresh) | (m['breadth'] < 0.3)).astype(int)
    return m

# ============================================================================
# SESSION STATE & DASHBOARD
# ============================================================================
if 'test_df' not in st.session_state: st.session_state.test_df = None

if run_analysis and selected_stocks:
    with st.spinner("Analyzing Market Patterns..."):
        raw = fetch_data(selected_stocks, start_date, end_date)
        if raw:
            processed = pd.concat([create_features(raw[t], t) for t in raw]).sort_index()
            st.session_state.test_df = detect_anomalies(processed, ret_threshold, vol_threshold, range_threshold)
            st.session_state.market_df = compute_market_metrics(st.session_state.test_df)
            st.success("✅ Done!")

if st.session_state.test_df is not None:
    test, market = st.session_state.test_df, st.session_state.market_df

    # 1. METRICS & VERDICT
    st.header("📊 Market Pulse")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading Days", len(test.index.unique()), help="Total days analyzed.")
    c2.metric("Anomalies", test['anomaly_flag'].sum(), f"{test['anomaly_flag'].mean()*100:.1f}% rate", help="Days exceeding your set thresholds.")
    c3.metric("Market Stress Days", market['market_anomaly_flag'].sum(), help="Systemic events where most stocks crashed.")
    c4.metric("Avg Breadth", f"{market['breadth'].mean()*100:.1f}%", help="% of stocks moving up. Lower = Bearish.")

    latest = market.iloc[-1]
    if latest['market_anomaly_flag']:
        st.error(f"🛑 **Status ({market.index[-1].date()}): High Stress.** Systemic anomalies detected.")
    else:
        st.success(f"✅ **Status ({market.index[-1].date()}): Stable.** Market behavior is within normal bounds.")

    # 2. DATE QUERY TOOL
    st.header("🔍 Investigate a Date")
    q_date = st.date_input("Query Date", value=test.index.max().date(), min_value=test.index.min().date(), max_value=test.index.max().date())
    q_str = str(q_date)
    
    if q_str in market.index.astype(str):
        m_day = market.loc[q_str]
        st.markdown(f"**Market Move:** {m_day['market_ret']*100:.2f}% | **Stocks Up:** {m_day['breadth']*100:.0f}%")
        
        day_anoms = test[test.index.astype(str) == q_str]
        flagged = day_anoms[day_anoms['anomaly_flag'] == 1]
        
        if not flagged.empty:
            for _, r in flagged.iterrows():
                reasons = []
                if abs(r['ret_z']) > ret_threshold: reasons.append(f"extreme price {'gain' if r['Return']>0 else 'drop'} ({r['Return']*100:.2f}%)")
                if r['volz'] > vol_threshold: reasons.append(f"unusual volume spike ({r['volz']:.1f}x normal)")
                st.write(f"🚩 **{r['Ticker']}**: Flagged for " + " and ".join(reasons) + ".")
            st.dataframe(flagged[['Ticker', 'type', 'Return', 'ret_z', 'volz']].style.format({'Return': '{:.2%}', 'ret_z': '{:.2f}', 'volz': '{:.2f}'}))
        else:
            st.write("✅ No individual anomalies detected on this day.")
    else:
        st.warning("No data for this date (Weekend/Holiday).")

    # 3. CHARTS
    st.divider()
    st.header("📈 Visual Analysis")
    with st.expander("❓ How to read these charts?"):
        st.write("- **Price Chart:** Red 'X' marks are anomalies. \n- **Z-Scores:** Green line (Price) or Orange line (Volume) crossing the red dash = Anomaly. \n- **Breadth:** Purple area below 0.3 (30%) = Market-wide panic.")
    
    chart_stock = st.selectbox("Select Stock to Plot", selected_stocks)
    s_data = test[test['Ticker'] == chart_stock]
    s_anoms = s_data[s_data['anomaly_flag'] == 1]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f'{chart_stock} Price', 'Statistical Indicators', 'Market Breadth'))
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['Price'], name='Price', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_anoms.index, y=s_anoms['Price'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')), row=1, col=1)
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['ret_z'], name='Price Z', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=s_data.index, y=s_data['volz'], name='Volume Z', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=ret_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-ret_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_trace(go.Scatter(x=market.index, y=market['breadth'], name='Breadth', fill='tozeroy', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=3, col=1)
    fig.update_layout(height=800, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # 4. EXPORT
    csv = test[test['anomaly_flag'] == 1].to_csv().encode('utf-8')
    st.download_button("📥 Download Anomalies Report (CSV)", csv, "market_report.csv", "text/csv", use_container_width=True)

else:
    st.info("👈 Use the sidebar to select stocks and click 'Run Analysis'.")
    st.image("https://via.placeholder.com/1000x300/f0f2f6/1f77b4?text=Select+Stocks+and+Run+Analysis+to+Start", use_container_width=True)

st.divider()
st.caption("Burhanuddin Udaipurwala")
