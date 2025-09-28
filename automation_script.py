import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
# Source for all NSE symbols
NSE_SYMBOLS_URL = 'https://raw.githubusercontent.com/aaquibladiwala/Pocket-Pivot/main/EQUITY_L.csv'
# Output file name (This is the input for the Streamlit app)
OUTPUT_FILENAME = 'nse_historical_data.csv' 
# Days of history required (Max for 200-day SMA + buffer)
DAYS_BACK = 720

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_symbols():
    """Fetches the list of NSE symbols."""
    try:
        df = pd.read_csv(NSE_SYMBOLS_URL)
        # We need the base symbol for market cap lookup
        symbols = df['SYMBOL'].tolist()
        # We need the yfinance ticker for data fetch
        tickers = [s + '.NS' for s in symbols]
        logging.info(f"Fetched {len(symbols)} symbols.")
        return dict(zip(symbols, tickers))
    except Exception as e:
        logging.error(f"Error reading symbols from CSV: {e}")
        return {}

def fetch_single_stock_data(symbol, ticker):
    """Fetches historical data and market cap for a single stock."""
    try:
        data = yf.download(ticker, period=f"{DAYS_BACK}d")
        if data.empty:
            return None
        
        # Fetch market cap separately (cannot be fetched reliably in bulk download)
        ticker_info = yf.Ticker(ticker).info
        market_cap = ticker_info.get('marketCap', None)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        data['Symbol'] = symbol
        data['Market_Cap'] = market_cap
        return data

    except Exception as e:
        logging.warning(f"Error fetching data for {symbol} ({ticker}): {e}")
        return None

def run_automation():
    """Main function to run the raw data pipeline."""
    symbol_to_ticker = fetch_symbols()
    
    if not symbol_to_ticker:
        logging.warning("No tickers found. Exiting.")
        return

    logging.info(f"Starting parallel fetch for {len(symbol_to_ticker)} tickers...")
    
    all_data_frames = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_stock_data, symbol, ticker): symbol for symbol, ticker in symbol_to_ticker.items()}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if df is not None:
                    all_data_frames.append(df)
            except Exception as e:
                logging.error(f"Error collecting result for {symbol}: {e}")

    if all_data_frames:
        df_output = pd.concat(all_data_frames, ignore_index=True)
        
        # Save to CSV
        df_output.to_csv(OUTPUT_FILENAME, index=False)
        logging.info(f"Successfully generated and saved {len(df_output)} historical records to {OUTPUT_FILENAME}")
    else:
        logging.warning("No valid historical data records were generated.")

if __name__ == "__main__":
    run_automation()
