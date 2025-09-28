import streamlit as st
import yfinance as yf
import pandas as pd
from tqdm import tqdm
import logging
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Set up logging for errors
logging.basicConfig(filename='errors.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

@st.cache_data
def fetch_nse_symbols(max_symbols=None):
    try:
        df = pd.read_csv('https://github.com/aaquibladiwala/Pocket-Pivot/blob/main/EQUITY_L.csv')
        symbols = df['SYMBOL'].tolist()
        if max_symbols:
            symbols = symbols[:max_symbols]
        st.write(f"Fetched {len(symbols)} NSE symbols from EQUITY_L.csv.")
        return symbols
    except Exception as e:
        logging.warning(f"Error reading EQUITY_L.csv: {e}")
        st.error(f"Error reading EQUITY_L.csv: {e}. Using sample list.")
        return ['RELIANCE', 'INFY', 'TCS', 'HDFCBANK', 'ICICIBANK']

def detect_pocket_pivot(df, symbol, lookback=10, sma_period=50, max_distance_52w_high=0.05, 
                        tightness_percentage=5.0, min_volume=10000, price_above_low_multiplier=1.10,
                        use_sma50=True, use_sma200=True, use_volume_surge=True, use_tightness=True,
                        use_52w_high=True, use_price_above_low=True):
    try:
        # Calculate SMAs
        df['SMA50'] = df['Close'].rolling(sma_period).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        
        # Identify down days (close < previous close)
        df['DownDay'] = df['Close'] < df['Close'].shift(1)
        
        # Latest data and lookback
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        lookback_data = df.iloc[-lookback-1:-1]
        
        # Skip if insufficient data or low liquidity
        if pd.isna(latest['SMA50']) or pd.isna(latest['SMA200']) or len(lookback_data) < lookback or latest['Volume'] < min_volume:
            return None
        
        # Pocket pivot conditions
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
        
        # Only return results for pocket pivot signals with all conditions
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

def fetch_stock_data(symbol, days_back):
    try:
        ticker = yf.Ticker(symbol + '.NS')
        data = ticker.history(period=f"{days_back}d")
        return symbol, data
    except Exception as e:
        logging.warning(f"Error fetching data for {symbol}: {e}")
        return symbol, None

def fetch_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol + '.NS')
        return ticker.info.get('marketCap', 'N/A')
    except Exception as e:
        logging.warning(f"Error fetching market cap for {symbol}: {e}")
        return 'N/A'

@st.cache_data
def screen_stocks(symbols, days_back=365, lookback=10, sma_period=50, max_distance_52w_high=0.05, 
                 tightness_percentage=5.0, min_volume=10000, price_above_low_multiplier=1.10,
                 use_sma50=True, use_sma200=True, use_volume_surge=True, use_tightness=True,
                 use_52w_high=True, use_price_above_low=True):
    results = []
    
    # Batch fetch data for all symbols
    st.write("Fetching data for all stocks in batch...")
    try:
        tickers = [s + '.NS' for s in symbols]
        data = yf.download(tickers, period=f"{days_back}d", group_by='ticker', threads=True)
    except Exception as e:
        logging.warning(f"Batch download failed: {e}. Falling back to individual fetches.")
        data = None
    
    # Process stocks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        if data is not None and not data.empty:
            # Use batched data
            for symbol in tqdm(symbols, desc="Processing stocks"):
                try:
                    if (symbol + '.NS') in data:
                        stock_data = data[symbol + '.NS'].dropna()
                        stock_data['Symbol'] = symbol
                        futures.append(executor.submit(detect_pocket_pivot, stock_data, symbol, 
                                                    lookback, sma_period, max_distance_52w_high, 
                                                    tightness_percentage, min_volume, price_above_low_multiplier,
                                                    use_sma50, use_sma200, use_volume_surge, use_tightness,
                                                    use_52w_high, use_price_above_low))
                except Exception as e:
                    logging.warning(f"Error accessing batch data for {symbol}: {e}")
        else:
            # Fallback to individual fetches
            futures = [executor.submit(fetch_stock_data, symbol, days_back) for symbol in symbols]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching stocks"):
                symbol, stock_data = future.result()
                if stock_data is not None and not stock_data.empty:
                    stock_data['Symbol'] = symbol
                    futures.append(executor.submit(detect_pocket_pivot, stock_data, symbol, 
                                                lookback, sma_period, max_distance_52w_high, 
                                                tightness_percentage, min_volume, price_above_low_multiplier,
                                                use_sma50, use_sma200, use_volume_surge, use_tightness,
                                                use_52w_high, use_price_above_low))
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing stocks"):
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

def main():
    st.title("📈 NSE Pocket Pivot Screener")
    st.markdown("Adjust filters and select conditions to screen for pocket pivot signals in Indian stocks. Results update in real-time.")

    # Sidebar for filter inputs
    st.sidebar.header("Filter Settings")
    max_symbols = st.sidebar.slider("Max Stocks to Scan", 50, 2000, 500, help="Limit for speed (500 takes ~3-5 mins)")
    days_back = st.sidebar.slider("Data Period (days)", 252, 730, 365, help="Historical data for analysis")
    lookback = st.sidebar.slider("Volume Lookback (days)", 5, 20, 10, help="Days to check for avg volume")
    sma_period = st.sidebar.slider("SMA Period (days)", 20, 100, 50, help="Period for 50-day SMA")
    max_distance_52w_high = st.sidebar.slider("Max Distance from 52W High (%)", 1.0, 10.0, 5.0, step=0.1, help="Price within X% of 52-week high") / 100
    tightness_percentage = st.sidebar.slider("10-Day Tightness (%)", 1.0, 10.0, 5.0, step=0.1, help="Max high/low range over 10 days")
    min_volume = st.sidebar.number_input("Min Volume", 1000, 100000, 10000, help="Minimum daily volume")
    price_above_low_multiplier = st.sidebar.slider("Price Above 10-Day Low Multiplier", 1.0, 1.5, 1.10, step=0.01, help="Price > X * 10-day low")

    # Sidebar for condition toggles
    st.sidebar.header("Select Conditions")
    use_sma50 = st.sidebar.checkbox("Price > 50-day SMA", value=True)
    use_sma200 = st.sidebar.checkbox("Price > 200-day SMA", value=True)
    use_volume_surge = st.sidebar.checkbox("Volume > Avg Volume in Lookback", value=True)
    use_tightness = st.sidebar.checkbox("10-Day Tightness", value=True)
    use_52w_high = st.sidebar.checkbox("Near 52-Week High", value=True)
    use_price_above_low = st.sidebar.checkbox("Price > Lookback Low * Multiplier", value=True)

    # Run screener on button click
    if st.sidebar.button("Run Screener"):
        with st.spinner("Screening stocks..."):
            nse_symbols = fetch_nse_symbols(max_symbols=max_symbols)
            df_results = screen_stocks(nse_symbols, days_back, lookback, sma_period, max_distance_52w_high, 
                                     tightness_percentage, min_volume, price_above_low_multiplier,
                                     use_sma50, use_sma200, use_volume_surge, use_tightness,
                                     use_52w_high, use_price_above_low)
        
        if df_results.empty:
            st.warning("No pocket pivot signals found. Try relaxing filters or increasing max stocks.")
            if len(nse_symbols) <= 5:
                st.error("Only sample list used. Ensure EQUITY_L.csv is in Downloads folder.")
        else:
            # Display results
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
        - Adjust filters and select conditions in the sidebar, then click "Run Screener" to see results.
        - **Runtime**: ~3-5 mins for 500 stocks. Set `Max Stocks to Scan` to 100 for ~30-60 secs.
        - **EQUITY_L.csv**: Ensure it's in your Downloads folder (download from nseindia.com).
        - **Filters**:
          - Core: Price > previous close, volume > avg volume in lookback (if enabled), price > 50-day SMA (if enabled).
          - Custom: Price > 200-day SMA, 10-day high/low ≤ tightness %, price within max distance of 52W high and ≥ 7% below, price > lookback low * multiplier (all toggleable).
          - Liquidity: Volume ≥ min volume.
        - **Market Cap**: Displayed in results (fetched via yfinance).
        - **Errors**: Check `errors.log` for issues (e.g., yfinance rate limits).
        - **Deploy**: Hosted on Streamlit Community Cloud for output-only viewing.
        """)

if __name__ == "__main__":
    main()
