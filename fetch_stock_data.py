import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

# Set up logging
logging.basicConfig(filename='fetch_errors.log', level=logging.WARNING, 
                    format='%(asctime)s - %(message)s')

# Default parameters
MAX_SYMBOLS = 2500  # Adjust as needed
DAYS_BACK = 500   # Fetch 1 year of data

# Placeholder for NSE holidays (update with 2025 holidays)
NSE_HOLIDAYS_2025 = [
    # Example: '2025-01-26', '2025-08-15', '2025-10-02'  # Republic Day, Independence Day, Gandhi Jayanti
]

def is_trading_day(date):
    """Check if a date is a trading day (Monday to Friday, not a holiday)."""
    if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
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

def fetch_nse_symbols(max_symbols=None):
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/aaquibladiwala/Pocket-Pivot/main/EQUITY_L.csv')
        symbols = df['SYMBOL'].tolist()
        if max_symbols:
            symbols = symbols[:max_symbols]
        print(f"Fetched {len(symbols)} NSE symbols from EQUITY_L.csv.")
        return symbols
    except Exception as e:
        logging.warning(f"Error reading EQUITY_L.csv: {e}")
        print(f"Error reading EQUITY_L.csv: {e}. Using sample list.")
        return ['RELIANCE', 'INFY', 'TCS', 'HDFCBANK', 'ICICIBANK']

def fetch_stock_data(symbols, days_back, end_date=None):
    try:
        # Set end date to latest trading day if not specified
        if end_date is None:
            end_date = get_latest_trading_day(datetime.now())
        start_date = end_date - timedelta(days=days_back)
        
        tickers = [s + '.NS' for s in symbols]
        data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'), group_by='ticker', threads=True)
        if data is None or data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Process data into a single DataFrame
        all_data = []
        for symbol in symbols:
            if (symbol + '.NS') in data:
                df = data[symbol + '.NS'].dropna().reset_index()
                df['symbol'] = symbol
                df['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df = df[['symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'last_updated']]
                df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'last_updated']
                df['date'] = df['date'].astype(str)
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No valid data processed.")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    except Exception as e:
        logging.warning(f"Error fetching data: {e}")
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fetch stock data and save to CSV.")
    parser.add_argument('--force', action='store_true', help="Force data fetch even on non-trading days")
    args = parser.parse_args()
    
    # Check if today is a trading day
    today = datetime.now()
    if not args.force and not is_trading_day(today):
        print("Today is not a trading day. Skipping data fetch. Use --force to override.")
        exit(0)
    
    # Fetch symbols and data
    symbols = fetch_nse_symbols(max_symbols=MAX_SYMBOLS)
    df = fetch_stock_data(symbols, days_back=DAYS_BACK)
    
    if not df.empty:
        output_file = "stock_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Stock data for {len(symbols)} symbols saved to {output_file}")
    else:
        print("No data fetched. Check fetch_errors.log for details.")
