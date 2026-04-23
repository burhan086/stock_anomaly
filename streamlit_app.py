import streamlit as st
import numpy as np
import pandas as pd

# Try importing yfinance, install if missing
try:
    import yfinance as yf
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
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
    
    # Stock selection
    st.subheader("📊 Select Stocks")
    available_stocks = ['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA']
    selected_stocks = st.multiselect(
        "Choose stocks to analyze",
        available_stocks,
        default=['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
    )
    
    # Date range
    st.subheader("📅 Date Range")
    start_date = st.date_input("Start Date", value=pd.to_datetime('2018-01-01'))
    end_date = st.date_input("End Date", value=pd.to_datetime('2020-04-01'))
    
    # Thresholds
    st.subheader("🎚️ Detection Thresholds")
    ret_threshold = st.slider("Return Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    vol_threshold = st.slider("Volume Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    range_threshold = st.slider("Range Percentile Threshold", 90, 99, 95, 1)
    
    # Run button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("💡 Tip: Lower thresholds = more sensitive detection")

# ============================================================================
# CORE FUNCTIONS (Cached for speed)
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    """Download stock data"""
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        data[t] = df[['Open','High','Low','Close','Adj Close','Volume']]
    return data

def create_features(df, ticker):
    """Engineer features"""
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    df['LogVol'] = np.log(df['Volume'].replace(0, 1))
    
    # Z-scores
    ret_mean = df['Return'].shift(1).rolling(63).mean()
    ret_std = df['Return'].shift(1).rolling(63).std()
    df['ret_z'] = (df['Return'] - ret_mean) / ret_std
    
    vol_mean = df['LogVol'].shift(1).rolling(21).mean()
    vol_std = df['LogVol'].shift(1).rolling(21).std()
    df['volz'] = (df['LogVol'] - vol_mean) / vol_std
    
    # Range percentile
    rng = (df['High'] - df['Low']) / df['Close']
    df['range_pct'] = rng.rolling(63).rank(pct=True) * 100
    
    df['Ticker'] = ticker
    df['Price'] = df['Adj Close']
    return df.dropna()

def detect_anomalies(df, ret_thresh, vol_thresh, range_thresh):
    """Rule-based detection"""
    df['anomaly_flag'] = (
        (np.abs(df['ret_z']) > ret_thresh) | 
        (df['volz'] > vol_thresh) | 
        (df['range_pct'] > range_thresh)
    ).astype(int)
    
    def label_type(row):
        if not row['anomaly_flag']:
            return "Normal"
        types = []
        if np.abs(row['ret_z']) > ret_thresh:
            types.append("crash" if row['Return'] < 0 else "spike")
        if row['volz'] > vol_thresh:
            types.append("volume_shock")
        return "+".join(types) if types else "anomaly"
    
    df['type'] = df.apply(label_type, axis=1)
    return df

def compute_market_metrics(df):
    """Market-level aggregation"""
    # Group by date and calculate metrics
    market = df.groupby(df.index).agg({
        'Return': 'mean',
        'anomaly_flag': 'mean'
    })
    market.columns = ['market_ret', 'flag_rate']
    
    # Calculate breadth separately
    breadth_data = []
    for date in df.index.unique():
        day_data = df.loc[date]
        breadth = (day_data['Return'] > 0).sum() / len(day_data) if len(day_data) > 0 else 0
        breadth_data.append({'Date': date, 'breadth': breadth})
    
    breadth_df = pd.DataFrame(breadth_data).set_index('Date')
    market = market.join(breadth_df)
    
    # Market anomaly flag
    market['market_anomaly_flag'] = (
        (market['market_ret'].abs() > market['market_ret'].abs().quantile(0.95)) |
        (market['breadth'] < 0.3)
    ).astype(int)
    
    return market

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

if run_analysis and len(selected_stocks) > 0:
    
    with st.spinner("📥 Fetching stock data..."):
        data = fetch_data(selected_stocks, start_date, end_date)
    
    with st.spinner("🔧 Engineering features..."):
        all_df = pd.concat([create_features(data[t], t) for t in selected_stocks]).sort_index()
    
    with st.spinner("🔍 Detecting anomalies..."):
        test = all_df.loc[str(end_date.year - 1):str(end_date)].copy()
        test = detect_anomalies(test, ret_threshold, vol_threshold, range_threshold)
        market = compute_market_metrics(test)
    
    # Success message
    st.success("✅ Analysis Complete!")
    
    # ========================================================================
    # OVERVIEW METRICS
    # ========================================================================
    
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Trading Days",
            value=f"{len(test.index.unique()):,}",
            delta=None
        )
    
    with col2:
        anomaly_count = test['anomaly_flag'].sum()
        anomaly_rate = anomaly_count / len(test) * 100
        st.metric(
            label="Anomalies Detected",
            value=f"{anomaly_count}",
            delta=f"{anomaly_rate:.1f}% of days"
        )
    
    with col3:
        market_stress = market['market_anomaly_flag'].sum()
        st.metric(
            label="Market Stress Days",
            value=f"{market_stress}",
            delta="Market-wide events"
        )
    
    with col4:
        avg_breadth = market['breadth'].mean() * 100
        st.metric(
            label="Avg Market Breadth",
            value=f"{avg_breadth:.1f}%",
            delta="Stocks going up"
        )
    
    st.divider()
    
    # ========================================================================
    # DATE QUERY
    # ========================================================================
    
    st.header("🔍 Date Query Tool")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        query_date = st.date_input(
            "Select a date to investigate",
            value=test.index.max().date(),
            min_value=test.index.min().date(),
            max_value=test.index.max().date()
        )
    
    with col2:
        if st.button("🔎 Query Date", use_container_width=True):
            query_str = str(query_date)
            
            if query_str in market.index.astype(str):
                m = market.loc[query_str]
                day_data = test.loc[query_str]
                anoms = day_data[day_data['anomaly_flag'] == 1]
                
                # Market status
                if m['market_anomaly_flag']:
                    st.markdown(f"""
                    <div class="anomaly-alert">
                        <h4>🚨 MARKET ANOMALY DETECTED</h4>
                        <p><b>Market Return:</b> {m['market_ret']*100:.2f}%</p>
                        <p><b>Breadth:</b> {m['breadth']*100:.0f}% (stocks going up)</p>
                        <p><b>Flag Rate:</b> {m['flag_rate']*100:.0f}% (stocks flagged)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"✅ **Normal Market Day** | Return: {m['market_ret']*100:.2f}% | Breadth: {m['breadth']*100:.0f}%")
                
                # Stock anomalies
                if not anoms.empty:
                    st.subheader(f"🚨 Anomalous Stocks ({len(anoms)})")
                    
                    anom_df = anoms[['Ticker', 'type', 'Return', 'ret_z', 'volz']].copy()
                    anom_df['Return'] = anom_df['Return'] * 100
                    anom_df.columns = ['Ticker', 'Type', 'Return (%)', 'Return Z-Score', 'Volume Z-Score']
                    
                    st.dataframe(
                        anom_df.style.format({
                            'Return (%)': '{:.2f}',
                            'Return Z-Score': '{:.2f}',
                            'Volume Z-Score': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.success("✅ No stock anomalies detected on this date")
            else:
                st.warning("⚠️ No data available for selected date")
    
    st.divider()
    
    # ========================================================================
    # INTERACTIVE CHARTS
    # ========================================================================
    
    st.header("📈 Interactive Charts")
    
    # Stock selector for charts
    chart_stock = st.selectbox("Select stock to visualize", selected_stocks)
    
    stock_data = test[test['Ticker'] == chart_stock]
    anoms = stock_data[stock_data['anomaly_flag'] == 1]
    
    # Create 3-panel chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f'{chart_stock} - Price & Anomalies',
            'Z-Score Indicators',
            'Market Breadth'
        )
    )
    
    # Panel 1: Price
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['Price'],
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=anoms.index,
            y=anoms['Price'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ),
        row=1, col=1
    )
    
    # Panel 2: Z-scores
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['ret_z'],
            name='Return Z-Score',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['volz'],
            name='Volume Z-Score',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=ret_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-ret_threshold, line_dash="dash", line_color="red", row=2, col=1)
    
    # Panel 3: Breadth
    fig.add_trace(
        go.Scatter(
            x=market.index,
            y=market['breadth'],
            name='Market Breadth',
            fill='tozeroy',
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", row=3, col=1)
    
    # Layout
    fig.update_layout(
        height=900,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_yaxes(title_text="Breadth", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ========================================================================
    # ANOMALY TABLE
    # ========================================================================
    
    st.header("📋 All Detected Anomalies")
    
    anomaly_df = test[test['anomaly_flag'] == 1].reset_index()[
        ['Date', 'Ticker', 'type', 'Return', 'ret_z', 'volz', 'range_pct']
    ].copy()
    
    anomaly_df['Return'] = anomaly_df['Return'] * 100
    anomaly_df.columns = ['Date', 'Ticker', 'Type', 'Return (%)', 'Return Z', 'Volume Z', 'Range %']
    
    # Add filtering
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.multiselect(
            "Filter by type",
            options=anomaly_df['Type'].unique(),
            default=anomaly_df['Type'].unique()
        )
    
    with col2:
        filter_ticker = st.multiselect(
            "Filter by ticker",
            options=anomaly_df['Ticker'].unique(),
            default=anomaly_df['Ticker'].unique()
        )
    
    filtered_df = anomaly_df[
        (anomaly_df['Type'].isin(filter_type)) &
        (anomaly_df['Ticker'].isin(filter_ticker))
    ]
    
    st.dataframe(
        filtered_df.style.format({
            'Return (%)': '{:.2f}',
            'Return Z': '{:.2f}',
            'Volume Z': '{:.2f}',
            'Range %': '{:.0f}'
        }).background_gradient(subset=['Return (%)'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Anomalies CSV",
        data=csv,
        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    # ========================================================================
    # MONTHLY SUMMARY
    # ========================================================================
    
    st.header("📅 Monthly Summary")
    
    # Get available months
    test_months = test.index.to_period('M').unique().sort_values()
    
    selected_month = st.selectbox(
        "Select month to analyze",
        options=test_months.to_timestamp(),
        format_func=lambda x: x.strftime('%B %Y')
    )
    
    month_year = selected_month.to_period('M')
    month_data = test[test.index.to_period('M') == month_year]
    month_anoms = month_data[month_data['anomaly_flag'] == 1]
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trading Days", f"{len(month_data.index.unique())}")
    
    with col2:
        st.metric("Anomalies", f"{len(month_anoms)}")
    
    with col3:
        anomaly_pct = len(month_anoms) / len(month_data) * 100 if len(month_data) > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")
    
    # Top anomalies
    if not month_anoms.empty:
        st.subheader("Top 10 Anomalies")
        
        top10 = month_anoms.nlargest(10, 'ret_z')[
            ['Ticker', 'type', 'Return', 'ret_z', 'volz']
        ].reset_index()
        
        top10['Return'] = top10['Return'] * 100
        top10.columns = ['Date', 'Ticker', 'Type', 'Return (%)', 'Return Z', 'Volume Z']
        
        st.dataframe(
            top10.style.format({
                'Return (%)': '{:.2f}',
                'Return Z': '{:.2f}',
                'Volume Z': '{:.2f}'
            }),
            use_container_width=True
        )

else:
    # ========================================================================
    # WELCOME SCREEN (when no analysis run yet)
    # ========================================================================
    
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start")
    
    st.header("📚 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Detection Method")
        st.markdown("""
        **Rule-Based Thresholds:**
        - Return Z-Score > 2.5 → Extreme price move
        - Volume Z-Score > 2.5 → Unusual trading activity
        - Range > 95th percentile → High volatility
        
        **Anomaly Types:**
        - 🔴 **Crash**: Large negative return
        - 🟢 **Spike**: Large positive return
        - 🟡 **Volume Shock**: Unusual volume
        """)
    
    with col2:
        st.subheader("📊 Features")
        st.markdown("""
        **What You Get:**
        - ✅ Interactive date query tool
        - ✅ Real-time anomaly detection
        - ✅ Market-level stress indicators
        - ✅ Visual dashboards
        - ✅ Downloadable reports
        - ✅ Monthly summaries
        """)
    
    st.divider()
    
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1. **Select Stocks**: Choose 3-6 stocks from the sidebar
    2. **Set Date Range**: Recommended 2018-2020 for COVID crash analysis
    3. **Adjust Thresholds**: Default 2.5 works well (lower = more sensitive)
    4. **Run Analysis**: Click the blue button
    5. **Explore Results**: Use date query, charts, and tables
    """)
    
    # Example screenshot placeholder
    st.image("https://via.placeholder.com/1200x400/1f77b4/ffffff?text=Stock+Market+Anomaly+Detection+Dashboard", 
             caption="Dashboard Preview")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("📈 Stock Market Anomaly Detector | Burhanuddin Udaipurwala")
