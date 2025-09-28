import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from tqdm import tqdm
import logging
from tabulate import tabulate 
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime, timedelta
import os 

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore')

# Set up logging for errors
logging.basicConfig(filename='errors.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

# SQLite database setup
DB_NAME = "stock_data.db"

# Placeholder for NSE holidays 
NSE_HOLIDAYS_2025 = [
    # Example: '2025-01-26', '2025-08-15', '2025-10-02'
]

# --- Helper Functions (Trading Days, DB Init) ---

def is_trading_day(date):
    """Check if a date is a trading day (Monday to Friday, not a holiday)."""
    if date.weekday() >= 5:
        return False
    if date.strftime('%Y-%m-%d') in NSE_HOLIDAYS_2025:
        return False
    return True

def get_latest_trading_day(current_date):
    """Get the most recent trading day before or on the current date."""
    date = current_date
    while not is_trading_day(date):
        date -= timedelta(days=1)
    return date

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data
                (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, last_updated TEXT,
                 PRIMARY KEY (symbol, date))''')
    try:
        c.execute("ALTER TABLE stock_data ADD COLUMN last_updated TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def get_db_last_updated():
    """Get the most recent last_updated timestamp from stock_data."""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = "SELECT MAX(last_updated) AS last_updated FROM stock_data"
        result = pd.read_sql_query(query, conn)
        conn.close()
        last_updated = result['last_updated'].iloc[0]
        if last_updated:
            return pd.to_datetime(last_updated).strftime('%Y-%m-%d %H:%M:%S')
        return "Database not yet initialized"
    except Exception as e:
        logging.warning(f"Error fetching last updated timestamp: {e}")
        return "Database not yet initialized"

@st.cache_data
def fetch_nse_symbols(max_symbols=None):
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/aaquibladiwala/Pocket-Pivot/main/EQUITY_L.csv')
        symbols = df['SYMBOL'].tolist()
        if max_symbols:
            symbols = symbols[:max_symbols]
        return symbols
    except Exception as e:
        logging.warning(f"Error reading EQUITY_L.csv: {e}")
        st.error(f"Error reading EQUITY_L.csv: {e}. Using sample list.")
        return ['RELIANCE', 'INFY', 'TCS', 'HDFCBANK', 'ICICIBANK']

def fetch_cached_data(symbol, days_back):
    conn = sqlite3.connect(DB_NAME)
    query = f"SELECT * FROM stock_data WHERE symbol = ? AND date >= ?"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    df = pd.read_sql_query(query, conn, params=(symbol, start_date.strftime('%Y-%m-%d')))
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = [col.capitalize() for col in df.columns]
        df['Symbol'] = symbol
    return df

def save_to_cache(symbol, data):
    if data is None or data.empty:
        return
    conn = sqlite3.connect(DB_NAME)
    data = data.reset_index()
    data['symbol'] = symbol
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = data[['symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'last_updated']]
    data.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'last_updated']
    data['date'] = data['date'].astype(str)
    
    def insert_or_replace(conn, table, data_frame):
        cols = ', '.join(list(data_frame.columns))
        placeholders = ', '.join(['?'] * len(data_frame.columns))
        sql = f'INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})'
        c = conn.cursor()
        for row in data_frame.itertuples(index=False):
            c.execute(sql, row)
        conn.commit()

    try:
        insert_or_replace(conn, 'stock_data', data)
    except Exception as e:
        logging.warning(f"Error during INSERT OR REPLACE for {symbol}: {e}")
    finally:
        conn.close()

def fetch_stock_data(symbol, days_back):
    try:
        cached_data = fetch_cached_data(symbol, days_back)
        latest_date = cached_data.index.max() if not cached_data.empty else None
        end_date = datetime.now()
        latest_trading_day = get_latest_trading_day(end_date).date()
        
        needs_update = cached_data.empty or (latest_date.date() < latest_trading_day if latest_date else True)
        
        if needs_update:
            ticker = yf.Ticker(symbol + '.NS')
            data = ticker.history(period=f"{days_back}d")
            if not data.empty:
                save_to_cache(symbol, data)
                return symbol, data
        return symbol, cached_data
    except Exception as e:
        logging.warning(f"Error fetching data for {symbol}: {e}")
        return symbol, None

def detect_pocket_pivot(df, symbol, lookback=10, sma_period=50, max_distance_52w_high=0.05, 
                        tightness_percentage=5.0, min_volume=10000, price_above_low_multiplier=1.10,
                        use_sma50=True, use_sma200=True, use_volume_surge=True, use_tightness=True,
                        use_52w_high=True, use_price_above_low=True):
    try:
        # Calculate SMAs
        df['SMA50'] = df['Close'].rolling(sma_period).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        
        # Ensure enough data
        required_bars = max(200, lookback) + 2 
        if len(df) < required_bars:
            return None 

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        lookback_data = df.iloc[-lookback-1:-1]
        
        # Skip if insufficient analysis data or low liquidity
        if pd.isna(latest['SMA50']) or pd.isna(latest['SMA200']) or latest['Volume'] < min_volume:
            return None
        
        # Pocket pivot core conditions
        price_up = latest['Close'] > prev['Close']
        above_sma50 = latest['Close'] > latest['SMA50'] if use_sma50 else True
        above_sma200 = latest['Close'] > latest['SMA200'] if use_sma200 else True
        near_sma10 = abs(latest['Close'] - latest['SMA10']) / latest['Close'] < 0.02
        
        # Volume: Compare to average volume in lookback
        volume_condition = latest['Volume'] > lookback_data['Volume'].mean() if use_volume_surge and not lookback_data.empty else True
        
        # Tightness: High/low range <= tightness_percentage
        max_high = lookback_data['High'].max()
        min_low = lookback_data['Low'].min()
        tightness_condition = (max_high / min_low) <= (1 + tightness_percentage / 100) if use_tightness and min_low > 0 else True
        
        # 52-week high: Price within max_distance_52w_high and not below 7% from high
        max_52w_high = df['High'].rolling(252).max().iloc[-1]
        high_distance_condition = (0.93 <= latest['Close'] / max_52w_high <= (1.0 + max_distance_52w_high)) if use_52w_high and max_52w_high > 0 else True
        
        # Price > lookback low * price_above_low_multiplier
        min_10d_low = lookback_data['Low'].min()
        price_above_low_condition = latest['Close'] > min_10d_low * price_above_low_multiplier if use_price_above_low and min_10d_low > 0 else True
        
        # Final Signal Check
        if (price_up and above_sma50 and above_sma200 and volume_condition and 
            tightness_condition and high_distance_condition and price_above_low_condition):
            signal = "Pocket Pivot"
            if near_sma10:
                signal += " (Near 10-day SMA)"
            return {
                'Symbol': symbol,
                'Price': round(latest['Close'], 2),
                'SMA50': round(latest['SMA50'], 2),
                'SMA200': round(latest['SMA200'], 2),
                'Volume': int(latest['Volume']),
                'Signal': signal,
                'Near 10-day SMA': 'Yes' if near_sma10 else 'No',
                '52W High Distance (%)': round((latest['Close'] / max_52w_high - 1) * 100, 2),
                '10D Tightness (%)': round((max_high / min_low - 1) * 100, 2) if min_low > 0 else 'N/A',
                # Also save the raw metrics used for filtering later
                'Passes_Volume_Surge': volume_condition,
                'Passes_Tightness': tightness_condition,
                'Passes_52W_High': high_distance_condition,
                'Passes_Price_Above_Low': price_above_low_condition
            }
        return None
    except Exception as e:
        logging.warning(f"Error processing {symbol}: {e}")
        return None

def fetch_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol + '.NS')
        return ticker.info.get('marketCap', 'N/A')
    except Exception as e:
        logging.warning(f"Error fetching market cap for {symbol}: {e}")
        return 'N/A'

@st.cache_data(show_spinner=False)
def screen_stocks(symbols, days_back=365, lookback=10, sma_period=50, max_distance_52w_high=0.05, 
                  tightness_percentage=5.0, min_volume=10000, price_above_low_multiplier=1.10,
                  use_sma50=True, use_sma200=True, use_volume_surge=True, use_tightness=True,
                  use_52w_high=True, use_price_above_low=True):
    
    results = []
    analysis_futures = []

    st.write(f"Fetching data for {len(symbols)} stocks...")
    try:
        tickers = [s + '.NS' for s in symbols]
        data_all = yf.download(tickers, period=f"{days_back}d", group_by='ticker', threads=True)
    except Exception as e:
        logging.warning(f"Batch download failed: {e}. Falling back to individual fetches.")
        data_all = None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        if data_all is not None and not data_all.empty:
            for symbol in tqdm(symbols, desc="Processing stocks (Batch Download)"):
                try:
                    ticker_key = symbol + '.NS'
                    if ticker_key in data_all.columns.get_level_values(0) or ticker_key in data_all.columns:
                        stock_data = data_all[ticker_key].dropna()
                        
                        if not stock_data.empty:
                            if isinstance(stock_data.columns, pd.MultiIndex):
                                stock_data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                            stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                            
                            save_to_cache(symbol, stock_data)
                            
                            analysis_futures.append(executor.submit(detect_pocket_pivot, stock_data.copy(), symbol, 
                                                                    lookback, sma_period, max_distance_52w_high, 
                                                                    tightness_percentage, min_volume, price_above_low_multiplier,
                                                                    use_sma50, use_sma200, use_volume_surge, use_tightness,
                                                                    use_52w_high, use_price_above_low))
                except Exception as e:
                    logging.warning(f"Error accessing batch data for {symbol}: {e}")
        else:
            st.write("Falling back to individual fetches (slower, but uses cache)...")
            fetch_futures = [executor.submit(fetch_stock_data, symbol, days_back) for symbol in symbols]
            
            data_to_analyze = []
            for future in tqdm(as_completed(fetch_futures), total=len(fetch_futures), desc="Fetching stocks (Individual)"):
                symbol, stock_data = future.result()
                if stock_data is not None and not stock_data.empty:
                    data_to_analyze.append((symbol, stock_data))

            st.write(f"Submitting analysis for {len(data_to_analyze)} stocks...")
            analysis_futures = [executor.submit(detect_pocket_pivot, stock_data.copy(), symbol, 
                                                lookback, sma_period, max_distance_52w_high, 
                                                tightness_percentage, min_volume, price_above_low_multiplier,
                                                use_sma50, use_sma200, use_volume_surge, use_tightness,
                                                use_52w_high, use_price_above_low) 
                                for symbol, stock_data in data_to_analyze]
        
        for future in tqdm(as_completed(analysis_futures), total=len(analysis_futures), desc="Analyzing stocks"):
            result = future.result()
            if result:
                results.append(result)

    df_results = pd.DataFrame(results)
    
    # Fetch market cap for results
    if not df_results.empty:
        st.write("Fetching market caps for signals...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            market_caps = list(executor.map(fetch_market_cap, df_results['Symbol']))
        df_results['Market Cap'] = market_caps
    
    return df_results


# --- NEW FILTERING FUNCTION ---

def filter_results(df, max_distance_52w_high_decimal, tightness_percentage, 
                   min_volume, use_sma50, use_sma200, use_tightness, use_52w_high):
    """
    Filters the pre-calculated Pocket Pivot results based on current user parameters 
    in memory (instantaneous).
    """
    if df.empty:
        return pd.DataFrame()

    df_filtered = df.copy()

    # 1. Volume (Liquidity) Filter
    df_filtered = df_filtered[df_filtered['Volume'] >= min_volume]
    
    # 2. SMA Conditions (Price vs. SMA)
    # The 'Price > Close' condition is implicit in the full PP signal and cannot be easily filtered here.
    if use_sma50:
        df_filtered = df_filtered[df_filtered['Price'] > df_filtered['SMA50']]
    if use_sma200:
        df_filtered = df_filtered[df_filtered['Price'] > df_filtered['SMA200']]
        
    # 3. Tightness Condition
    if use_tightness:
        # 10D Tightness (%) column is stored as percentage
        df_filtered = df_filtered[df_filtered['10D Tightness (%)'] <= tightness_percentage]

    # 4. 52W High Distance
    if use_52w_high:
        max_dist_percent = max_distance_52w_high_decimal * 100
        # Filter for price within the max_dist_percent of the 52W high
        df_filtered = df_filtered[(df_filtered['52W High Distance (%)'] <= max_dist_percent)]
        
    # NOTE: Volume Surge and Price Above Low filters rely on the 'Signal' being present 
    #       in the initial run, as these are harder to re-calculate without the raw data.
    #       We rely on the initial run's parameters for these.
        
    return df_filtered.reset_index(drop=True)

# --- Session State Initialization ---

def initialize_session_state():
    """Initializes session state keys for persistent filters and data cache."""
    if 'max_symbols' not in st.session_state:
        st.session_state.max_symbols = 500
    if 'days_back' not in st.session_state:
        st.session_state.days_back = 365
    if 'lookback' not in st.session_state:
        st.session_state.lookback = 10
    if 'sma_period' not in st.session_state:
        st.session_state.sma_period = 50
    if 'max_distance_52w_high_percent' not in st.session_state:
        st.session_state.max_distance_52w_high_percent = 5.0
    if 'tightness_percentage' not in st.session_state:
        st.session_state.tightness_percentage = 5.0
    if 'min_volume' not in st.session_state:
        st.session_state.min_volume = 10000
    if 'price_above_low_multiplier' not in st.session_state:
        st.session_state.price_above_low_multiplier = 1.10
    if 'use_sma50' not in st.session_state:
        st.session_state.use_sma50 = True
    if 'use_sma200' not in st.session_state:
        st.session_state.use_sma200 = True
    if 'use_volume_surge' not in st.session_state:
        st.session_state.use_volume_surge = True
    if 'use_tightness' not in st.session_state:
        st.session_state.use_tightness = True
    if 'use_52w_high' not in st.session_state:
        st.session_state.use_52w_high = True
    if 'use_price_above_low' not in st.session_state:
        st.session_state.use_price_above_low = True
        
    # Persistent Data Caching for the "Load Once, Filter Many" model
    if 'df_full_results_cache' not in st.session_state:
        st.session_state.df_full_results_cache = pd.DataFrame() 
    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame()
    if 'initial_results_loaded' not in st.session_state:
        st.session_state.initial_results_loaded = False
    
# --- MAIN APPLICATION FUNCTION ---

def main():
    initialize_session_state()
    init_db()
    
    st.title("📈 NSE Pocket Pivot Screener (Instant Filtering)")
    st.markdown("Data is loaded and analyzed **once** upon startup for speed. Changing parameters instantly **filters** the existing results.")

    st.write(f"**Database Last Updated**: {get_db_last_updated()}")

    # Sidebar setup
    st.sidebar.header("Filter Settings")
    
    manual_data_refresh = st.sidebar.button("Force Full Data Refresh (Long)")
    
    # --- WIDGETS (linked directly to session state keys) ---
    st.sidebar.slider("Max Stocks to Scan", 50, 2000, st.session_state.max_symbols, 
                      key='max_symbols', help="Only affects initial data load (use 2000 for full NSE)")
    st.sidebar.slider("Data Period (days)", 252, 730, st.session_state.days_back, 
                      key='days_back', help="Historical data for analysis")
    st.sidebar.slider("Volume Lookback (days)", 5, 20, st.session_state.lookback, 
                      key='lookback', help="Days to check for avg volume")
    st.sidebar.slider("SMA Period (days)", 20, 100, st.session_state.sma_period, 
                      key='sma_period', help="Period for 50-day SMA")
    st.sidebar.slider("Max Distance from 52W High (%)", 1.0, 10.0, st.session_state.max_distance_52w_high_percent, step=0.1, 
                      key='max_distance_52w_high_percent', help="Price within X% of 52-week high")
    st.sidebar.slider("10-Day Tightness (%)", 1.0, 10.0, st.session_state.tightness_percentage, step=0.1, 
                      key='tightness_percentage', help="Max high/low range over 10 days")
    st.sidebar.number_input("Min Volume", 1000, 100000, st.session_state.min_volume, 
                            key='min_volume', help="Minimum daily volume")
    st.sidebar.slider("Price Above 10-Day Low Multiplier", 1.0, 1.5, st.session_state.price_above_low_multiplier, step=0.01, 
                      key='price_above_low_multiplier', help="This impacts the initial *analysis*, not the filter.")

    st.sidebar.header("Select Conditions")
    st.sidebar.checkbox("Price > 50-day SMA", value=st.session_state.use_sma50, key='use_sma50')
    st.sidebar.checkbox("Price > 200-day SMA", value=st.session_state.use_sma200, key='use_sma200')
    st.sidebar.checkbox("Volume > Avg Volume in Lookback (Initial Check)", value=st.session_state.use_volume_surge, key='use_volume_surge')
    st.sidebar.checkbox("10-Day Tightness", value=st.session_state.use_tightness, key='use_tightness')
    st.sidebar.checkbox("Near 52-Week High", value=st.session_state.use_52w_high, key='use_52w_high')
    st.sidebar.checkbox("Price > Lookback Low * Multiplier (Initial Check)", value=st.session_state.use_price_above_low, key='use_price_above_low')

    
    # --- CORE LOGIC: ONE-TIME DATA LOAD / FILTERING ---
    
    # Condition to run the heavy screen_stocks function: first load or manual refresh
    if not st.session_state.initial_results_loaded or manual_data_refresh:
        
        st.warning("Initial full data analysis running... This will take a few minutes. Subsequent filtering will be instant.")
        
        # Convert percentage slider value to decimal for the function
        max_dist_decimal = st.session_state.max_distance_52w_high_percent / 100.0

        if manual_data_refresh:
            st.cache_data.clear()

        with st.spinner(f"Running full analysis on {st.session_state.max_symbols} stocks..."):
            nse_symbols = fetch_nse_symbols(max_symbols=st.session_state.max_symbols)
            df_full_results = screen_stocks(
                symbols=nse_symbols,
                days_back=st.session_state.days_back,
                lookback=st.session_state.lookback,
                sma_period=st.session_state.sma_period,
                max_distance_52w_high=max_dist_decimal, 
                tightness_percentage=st.session_state.tightness_percentage,
                min_volume=st.session_state.min_volume,
                price_above_low_multiplier=st.session_state.price_above_low_multiplier,
                use_sma50=st.session_state.use_sma50,
                use_sma200=st.session_state.use_sma200,
                use_volume_surge=st.session_state.use_volume_surge,
                use_tightness=st.session_state.use_tightness,
                use_52w_high=st.session_state.use_52w_high,
                use_price_above_low=st.session_state.use_price_above_low
            )
            # Store the complete, initial result set in session state
            st.session_state.df_full_results_cache = df_full_results.copy()
            st.session_state.initial_results_loaded = True

    
    # --- INSTANT FILTERING LOGIC (Runs on every app rerun/widget change) ---
    if st.session_state.initial_results_loaded:
        df_full = st.session_state.df_full_results_cache

        # Convert percentage slider value to decimal for the filter
        max_dist_decimal = st.session_state.max_distance_52w_high_percent / 100.0
        
        # Apply the quick, in-memory filter based on current sidebar settings
        df_filtered = filter_results(
            df=df_full,
            max_distance_52w_high_decimal=max_dist_decimal,
            tightness_percentage=st.session_state.tightness_percentage,
            min_volume=st.session_state.min_volume,
            use_sma50=st.session_state.use_sma50,
            use_sma200=st.session_state.use_sma200,
            use_tightness=st.session_state.use_tightness,
            use_52w_high=st.session_state.use_52w_high,
        )
        st.session_state.df_results = df_filtered

    
    # --- DISPLAY RESULTS ---
    df_results = st.session_state.df_results

    if df_results.empty and st.session_state.initial_results_loaded:
        st.warning("No pocket pivot signals found with current filters. Try relaxing the sidebar settings.")
    elif not df_results.empty:
        # Display the filtered data
        st.subheader(f"Pocket Pivot Signals ({len(df_results)} found)")
        st.dataframe(df_results[['Symbol', 'Price', 'SMA50', 'SMA200', 'Market Cap', 'Volume', 'Signal', 
                                 'Near 10-day SMA', '52W High Distance (%)', '10D Tightness (%)']], 
                     use_container_width=True)
        
        # Download CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "pocket_pivot_filtered_results.csv", "text/csv")
        
        # Summary
        st.metric("Total Pocket Pivot Signals", len(df_results))
        st.write(f"Last Full Analysis Run: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Instructions
    with st.expander("How to Use & Notes"):
        st.markdown("""
        - **Initial Load**: The app runs a heavy analysis on startup to generate all possible signals. This is the only slow part (a few minutes).
        - **Instant Filtering**: Changing any filter in the sidebar instantly applies the filter to the pre-calculated dataset in memory.
        - **Force Refresh**: Use the 'Force Full Data Refresh' button to fetch new data from yfinance and re-run the initial heavy analysis.
        - **Initial Checkboxes**: The `Volume Surge` and `Price Above Low Multiplier` checkboxes only affect the *initial* analysis run, as these values are used to determine the original Pocket Pivot signal. Other filters (`SMA`, `Tightness`, `52W High`, `Volume`) can be adjusted instantly.
        """)

if __name__ == "__main__":
    main()
