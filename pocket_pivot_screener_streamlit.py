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
import os # Added for robust file management

# Suppress pandas FutureWarnings inside yfinance
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
    # Create table with last_updated column
    c.execute('''CREATE TABLE IF NOT EXISTS stock_data
                (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, last_updated TEXT,
                 PRIMARY KEY (symbol, date))''')
    # Add last_updated column to existing table if missing
    try:
        c.execute("ALTER TABLE stock_data ADD COLUMN last_updated TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
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
        # st.write(f"Fetched {len(symbols)} NSE symbols from EQUITY_L.csv.") # Removed for cleaner UI
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
    
    # Use INSERT OR REPLACE via custom method for thread-safe UPSERT
    def insert_or_replace(conn, table, data_frame):
        cols = ', '.join(list(data_frame.columns))
        placeholders = ', '.join(['?'] * len(data_frame.columns))
        # SQL command using INSERT OR REPLACE (due to PRIMARY KEY on symbol, date)
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
        
        # Latest data and lookback
        # Ensure we have enough data (at least 200 bars + lookback + 1 for lookback_data)
        required_bars = max(200, lookback) + 2 
        if len(df) < required_bars:
            return None 

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        lookback_data = df.iloc[-lookback-1:-1]
        
        # Skip if insufficient data or low liquidity
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
                '10D Tightness (%)': round((max_high / min_low - 1) * 100, 2) if min_low > 0 else 'N/A'
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

    # Attempt Batch Fetch
    st.write(f"Fetching data for {len(symbols)} stocks...")
    try:
        tickers = [s + '.NS' for s in symbols]
        data_all = yf.download(tickers, period=f"{days_back}d", group_by='ticker', threads=True)
    except Exception as e:
        logging.warning(f"Batch download failed: {e}. Falling back to individual fetches.")
        data_all = None
    
    # Process stocks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        if data_all is not None and not data_all.empty:
            # STAGE 1: Process Batch Data
            for symbol in tqdm(symbols, desc="Processing stocks (Batch Download)"):
                try:
                    ticker_key = symbol + '.NS'
                    if ticker_key in data_all.columns.get_level_values(0):
                        stock_data = data_all[ticker_key].dropna()
                    elif ticker_key in data_all.columns: # for single ticker download
                         stock_data = data_all[ticker_key].dropna()
                    else:
                        continue
                        
                    if not stock_data.empty:
                        # Normalize columns (yf.download returns multi-level for multi-ticker)
                        # We only need Open, High, Low, Close, Volume
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
            # STAGE 1: Fallback to Individual Fetches
            st.write("Falling back to individual fetches (slower, but uses cache)...")
            fetch_futures = [executor.submit(fetch_stock_data, symbol, days_back) for symbol in symbols]
            
            data_to_analyze = []
            for future in tqdm(as_completed(fetch_futures), total=len(fetch_futures), desc="Fetching stocks (Individual)"):
                symbol, stock_data = future.result()
                if stock_data is not None and not stock_data.empty:
                    data_to_analyze.append((symbol, stock_data))

            # STAGE 2: Submit Analysis Tasks
            st.write(f"Submitting analysis for {len(data_to_analyze)} stocks...")
            analysis_futures = [executor.submit(detect_pocket_pivot, stock_data.copy(), symbol, 
                                                lookback, sma_period, max_distance_52w_high, 
                                                tightness_percentage, min_volume, price_above_low_multiplier,
                                                use_sma50, use_sma200, use_volume_surge, use_tightness,
                                                use_52w_high, use_price_above_low) 
                                for symbol, stock_data in data_to_analyze]
        
        # STAGE 3: Collect Analysis Results
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

# --- NEW: Function to initialize session state with default values ---
def initialize_session_state():
    """Initializes session state keys for persistent filters."""
    # Set default values for all inputs
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
    if 'df_results' not in st.session_state:
        st.session_state.df_results = pd.DataFrame() # Store the results dataframe
    if 'screener_ran' not in st.session_state:
        st.session_state.screener_ran = False
    
# --- REFACTORED MAIN FUNCTION ---
def main():
    initialize_session_state()
    init_db()
    
    st.title("📈 NSE Pocket Pivot Screener")
    st.markdown("Adjust filters and select conditions to screen for pocket pivot signals in Indian stocks. **Screening runs automatically** when filter settings change.")

    st.write(f"**Database Last Updated**: {get_db_last_updated()}")

    # Sidebar for filter inputs
    st.sidebar.header("Filter Settings")
    
    # Run Screener Button (used to manually force a run when filter values haven't changed)
    manual_run = st.sidebar.button("Force Run Screener (Clear Cache)")
    
    # --- CORRECTED WIDGETS: Using key= and default value= but NO REDUNDANT ASSIGNMENT ---

    st.sidebar.slider("Max Stocks to Scan", 50, 2000, st.session_state.max_symbols, 
                      key='max_symbols', help="Limit for speed (500 takes ~3-5 mins for first run, faster with cache)")
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
                      key='price_above_low_multiplier', help="Price > X * 10-day low")

    # Sidebar for condition toggles
    st.sidebar.header("Select Conditions")
    st.sidebar.checkbox("Price > 50-day SMA", value=st.session_state.use_sma50, key='use_sma50')
    st.sidebar.checkbox("Price > 200-day SMA", value=st.session_state.use_sma200, key='use_sma200')
    st.sidebar.checkbox("Volume > Avg Volume in Lookback", value=st.session_state.use_volume_surge, key='use_volume_surge')
    st.sidebar.checkbox("10-Day Tightness", value=st.session_state.use_tightness, key='use_tightness')
    st.sidebar.checkbox("Near 52-Week High", value=st.session_state.use_52w_high, key='use_52w_high')
    st.sidebar.checkbox("Price > Lookback Low * Multiplier", value=st.session_state.use_price_above_low, key='use_price_above_low')

    
    # --- AUTO-RUN LOGIC ---
    # The app reruns when any widget changes. We use st.session_state.screener_ran 
    # to prevent a full run on every initial page load after a filter change.
    
    # Check if any filter value has changed (Streamlit's core mechanism) or if forced
    if not st.session_state.screener_ran or manual_run:
        
        # Convert percentage slider value to decimal for the function
        max_dist_decimal = st.session_state.max_distance_52w_high_percent / 100.0

        if manual_run:
            # Clear the cache to ensure fresh data fetch on manual run
            st.cache_data.clear()
            st.warning("Data cache cleared. Running with fresh data fetches (may take longer).")

        with st.spinner(f"Screening {st.session_state.max_symbols} stocks..."):
            nse_symbols = fetch_nse_symbols(max_symbols=st.session_state.max_symbols)
            df_results = screen_stocks(
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
            st.session_state.df_results = df_results
            st.session_state.screener_ran = True

    # --- DISPLAY RESULTS ---
    df_results = st.session_state.df_results

    if df_results.empty and st.session_state.screener_ran:
        st.warning("No pocket pivot signals found. Try relaxing filters or increasing max stocks.")
        if st.session_state.max_symbols <= 5:
            st.error("Only sample list used. Ensure EQUITY_L.csv is accessible at the GitHub URL.")
    elif not df_results.empty:
        st.subheader("Pocket Pivot Signals")
        st.dataframe(df_results[['Symbol', 'Price', 'SMA50', 'SMA200', 'Market Cap', 'Volume', 'Signal', 
                                 'Near 10-day SMA', '52W High Distance (%)', '10D Tightness (%)']], 
                     use_container_width=True)
        
        # Download CSV
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "pocket_pivot_results.csv", "text/csv")
        
        # Summary
        st.metric("Total Pocket Pivot Signals", len(df_results))
        st.write(f"Scan completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Instructions
    with st.expander("How to Use & Notes"):
        st.markdown("""
        - **Auto-Run**: The screener runs automatically whenever you adjust a filter or toggle a condition in the sidebar.
        - **Force Run**: Use the 'Force Run Screener' button to clear the Streamlit internal cache and re-run on fresh data.
        - **Runtime**: The first run for 500 stocks takes about ~3-5 mins; subsequent runs are faster with cached data (~30-60 secs).
        - **Data Caching**: Stock data is efficiently cached in `stock_data.db` to minimize API calls.
        """)

if __name__ == "__main__":
    main()
