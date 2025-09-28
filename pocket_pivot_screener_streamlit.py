import streamlit as st
import pandas as pd
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')

# --- Configuration ---
# >>> CRITICAL: UPDATE THIS URL with the public link to your raw historical data CSV <<<
GITHUB_RAW_DATA_URL = 'https://raw.githubusercontent.com/aaquibladiwala/Pocket-Pivot/main/nse_historical_data.csv'

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')

# --- Session State and Utility Functions ---

def initialize_session_state():
    """Initializes session state keys for persistent filters."""
    if 'max_symbols' not in st.session_state: st.session_state.max_symbols = 2000
    if 'sma_period' not in st.session_state: st.session_state.sma_period = 50
    if 'lookback' not in st.session_state: st.session_state.lookback = 10 
    if 'max_distance_52w_high_percent' not in st.session_state: st.session_state.max_distance_52w_high_percent = 5.0
    if 'tightness_percentage' not in st.session_state: st.session_state.tightness_percentage = 5.0
    if 'min_volume' not in st.session_state: st.session_state.min_volume = 10000
    if 'use_sma50' not in st.session_state: st.session_state.use_sma50 = True
    if 'use_sma200' not in st.session_state: st.session_state.use_sma200 = True
    if 'use_volume_surge' not in st.session_state: st.session_state.use_volume_surge = True
    if 'use_tightness' not in st.session_state: st.session_state.use_tightness = True
    if 'use_52w_high' not in st.session_state: st.session_state.use_52w_high = True
        
    if 'df_raw_data' not in st.session_state: st.session_state.df_raw_data = pd.DataFrame() 
    if 'df_metrics_cache' not in st.session_state: st.session_state.df_metrics_cache = pd.DataFrame()
    if 'df_results' not in st.session_state: st.session_state.df_results = pd.DataFrame()
    if 'raw_data_loaded' not in st.session_state: st.session_state.raw_data_loaded = False
    
def get_data_snapshot_date(df_raw):
    """Retrieves the latest date from the loaded raw data for display."""
    if df_raw.empty:
        return "Data not yet loaded"
    try:
        return df_raw['Date'].max().strftime('%Y-%m-%d')
    except Exception:
        return "Date calculation error"


# --- DATA LOADING AND METRIC CALCULATION ---

@st.cache_data(show_spinner=False)
def load_raw_data_from_github():
    """Loads the multi-day historical data snapshot from GitHub instantly."""
    try:
        df = pd.read_csv(GITHUB_RAW_DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'])
        
        df.rename(columns={'Market_Cap': 'Market Cap', 'Close': 'Price'}, inplace=True)
        df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
        
        st.success(f"Successfully loaded raw historical data for {df['Symbol'].nunique()} stocks.")
        return df

    except Exception as e:
        st.error(f"Error loading raw data from GitHub. Check the URL/script output. Error: {e}")
        logging.error(f"GitHub raw data load error: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def calculate_all_metrics(df_raw, sma_period, lookback):
    """
    Calculates dynamic metrics (SMAs, Tightness, Volume Avg, PP Signal) 
    based on user-defined periods. This is the heavy, cached step.
    """
    if df_raw.empty:
        return pd.DataFrame()

    logging.info(f"Recalculating metrics with SMA={sma_period}, Lookback={lookback}...")

    grouped = df_raw.groupby('Symbol')
    results = []
    
    # Process each stock's historical data
    for symbol, df_stock in grouped:
        df_stock = df_stock.sort_values('Date').set_index('Date')
        
        # Ensure minimum bars for 200 SMA and lookback
        required_bars = max(200, lookback) + 2
        if len(df_stock) < required_bars:
            continue

        # 1. Calculate Dynamic Rolling Metrics
        df_stock['SMA50'] = df_stock['Price'].rolling(sma_period).mean()
        df_stock['SMA200'] = df_stock['Price'].rolling(200).mean()
        df_stock['Max_52W_High'] = df_stock['High'].rolling(252).max()
        
        # 2. Extract Latest Day Metrics
        latest = df_stock.iloc[-1]
        if pd.isna(latest['SMA50']) or pd.isna(latest['SMA200']):
            continue
            
        prev = df_stock.iloc[-2]
        lookback_data = df_stock.iloc[-lookback-1:-1]
        
        # 3. Tightness and Volume Metrics
        max_high = lookback_data['High'].max()
        min_low = lookback_data['Low'].min()
        
        tightness_pct = ((max_high / min_low) - 1) * 100 if min_low > 0 else float('inf')
        avg_lookback_volume = lookback_data['Volume'].mean()
        
        # 4. Final Analysis/Signal
        try:
            dist_52w_high = (latest['Price'] / latest['Max_52W_High'] - 1) * 100
        except ZeroDivisionError:
            dist_52w_high = float('inf')

        # Pocket Pivot Signal (must meet Price Up AND Volume Surge for base signal)
        signal = 'No Signal'
        if latest['Price'] > prev['Price']:
            if latest['Volume'] > avg_lookback_volume:
                signal = 'Pocket Pivot'
        
        results.append({
            'Symbol': symbol,
            'Price': round(latest['Price'], 2),
            'Volume': int(latest['Volume']),
            'Market Cap': latest['Market Cap'],
            'SMA50': round(latest['SMA50'], 2),
            'SMA200': round(latest['SMA200'], 2),
            '10D Tightness (%)': round(tightness_pct, 2),
            '52W High Distance (%)': round(dist_52w_high, 2),
            'Signal': signal
        })
        
    return pd.DataFrame(results)

def filter_results(df, max_distance_52w_high_decimal, tightness_percentage, 
                   min_volume, use_sma50, use_sma200, use_volume_surge, use_tightness, use_52w_high):
    """Filters the calculated metrics based on current user parameters instantly."""
    if df.empty:
        return pd.DataFrame()

    df_filtered = df.copy()
    
    # 1. Volume (Liquidity) Filter
    df_filtered = df_filtered[df_filtered['Volume'] >= min_volume]
    
    # 2. SMA Conditions 
    if use_sma50:
        df_filtered = df_filtered[df_filtered['Price'] > df_filtered['SMA50']]
    if use_sma200:
        df_filtered = df_filtered[df_filtered['Price'] > df_filtered['SMA200']]
        
    # 3. Tightness Condition (Dynamic)
    if use_tightness:
        df_filtered = df_filtered[df_filtered['10D Tightness (%)'] <= tightness_percentage]

    # 4. 52W High Distance (Dynamic)
    if use_52w_high:
        max_dist_percent = max_distance_52w_high_decimal * 100
        # Price must be near high AND not more than 7% below 52W high
        df_filtered = df_filtered[(df_filtered['52W High Distance (%)'] <= max_dist_percent) & 
                                  (df_filtered['52W High Distance (%)'] >= -7.0)]
        
    # 5. Volume Surge/Base Signal
    if use_volume_surge:
        df_filtered = df_filtered[df_filtered['Signal'] == 'Pocket Pivot']
        
    return df_filtered.reset_index(drop=True)

# --- MAIN APPLICATION FUNCTION ---

def main():
    initialize_session_state()
    
    st.title("📈 NSE Pocket Pivot Screener (Fully Dynamic)")
    st.markdown("Data is loaded once. **Changing periods (SMA/Lookback) triggers a fast recalculation** (via cache). Changing thresholds filters instantly.")

    st.write(f"**Data Snapshot Date**: {get_data_snapshot_date(st.session_state.df_raw_data)}")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.header("Calculation Periods")
        st.markdown("_Changing these busts the cache & recalculates metrics._")
        
        st.slider("SMA Period (days)", 20, 100, st.session_state.sma_period, 
                          key='sma_period', help="Period for the main moving average (e.g., 50)")
        st.slider("Tightness Lookback (days)", 5, 20, st.session_state.lookback, 
                          key='lookback', help="Window size for calculating price tightness.")
        
        st.header("Filter Thresholds")
        st.markdown("_Changing these filters the existing results instantly._")

        manual_data_refresh = st.button("Force Raw Data Reload (Slow)", help="Clears cache and forces a fresh download from GitHub.")

        st.slider("Max Distance from 52W High (%)", 1.0, 10.0, st.session_state.max_distance_52w_high_percent, step=0.1, 
                          key='max_distance_52w_high_percent', help="Price must be within X% of its 52-week high.")
        st.slider("Max 10D Tightness (%)", 1.0, 10.0, st.session_state.tightness_percentage, step=0.1, 
                          key='tightness_percentage', help="Maximum percentage range (High/Low) over the lookback period.")
        st.number_input("Min Volume", 1000, 100000, st.session_state.min_volume, 
                                key='min_volume', help="Minimum daily volume for liquidity.")

        st.header("Filter Conditions")
        st.checkbox("Price > Dynamic SMA", value=st.session_state.use_sma50, key='use_sma50')
        st.checkbox("Price > 200-day SMA", value=st.session_state.use_sma200, key='use_sma200')
        st.checkbox("Enforce Volume Surge (Dynamic)", value=st.session_state.use_volume_surge, key='use_volume_surge', help="Requires current volume > lookback average.")
        st.checkbox("Enforce Price Tightness (Dynamic)", value=st.session_state.use_tightness, key='use_tightness')
        st.checkbox("Enforce Near 52-Week High", value=st.session_state.use_52w_high, key='use_52w_high')
    
    # --- 1. Load Raw Data (Once or on manual refresh) ---
    if not st.session_state.raw_data_loaded or manual_data_refresh:
        
        if manual_data_refresh:
            st.cache_data.clear() 

        with st.spinner("1/2: Loading raw data from GitHub..."):
            df_raw = load_raw_data_from_github()
            st.session_state.df_raw_data = df_raw.copy()
            st.session_state.raw_data_loaded = True
            # Use st.rerun if data load changes state, ensuring metrics calc runs next
            if not df_raw.empty:
                st.rerun() 

    df_raw = st.session_state.df_raw_data
    if df_raw.empty:
        st.error("Cannot proceed with analysis. Please check data source and reload.")
        return

    # 2. Calculate Metrics (Cached based on SMA/Lookback periods)
    with st.spinner(f"2/2: Calculating metrics (SMA={st.session_state.sma_period}, Lookback={st.session_state.lookback})..."):
        # The arguments passed here (sma_period, lookback) determine if the cache is hit or recalculated.
        df_metrics_cache = calculate_all_metrics(
            df_raw, 
            st.session_state.sma_period, 
            st.session_state.lookback
        )
        st.session_state.df_metrics_cache = df_metrics_cache.copy()

    # 3. Instant Filtering (Runs instantly on every rerun with current filters)
    df_metrics = st.session_state.df_metrics_cache
    
    if df_metrics.empty:
        st.warning("Metric calculation returned no results. Check raw data history.")
        return

    df_filtered = filter_results(
        df=df_metrics,
        max_distance_52w_high_decimal=st.session_state.max_distance_52w_high_percent / 100,
        tightness_percentage=st.session_state.tightness_percentage,
        min_volume=st.session_state.min_volume,
        use_sma50=st.session_state.use_sma50,
        use_sma200=st.session_state.use_sma200,
        use_volume_surge=st.session_state.use_volume_surge,
        use_tightness=st.session_state.use_tightness,
        use_52w_high=st.session_state.use_52w_high
    )
    st.session_state.df_results = df_filtered

    
    # --- DISPLAY RESULTS ---
    df_results = st.session_state.df_results

    if df_results.empty:
        st.warning(f"No signals found with current filters and periods.")
    elif not df_results.empty:
        st.subheader(f"Pocket Pivot Signals ({len(df_results)} found)")
        st.dataframe(df_results[['Symbol', 'Price', 'SMA50', 'SMA200', 'Market Cap', 'Volume', 
                                 '10D Tightness (%)', '52W High Distance (%)', 'Signal']], 
                     use_container_width=True)
        
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "pocket_pivot_filtered_results.csv", "text/csv")
        
        st.metric("Total Signals Found", len(df_results))
        st.write(f"Last Filter Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
