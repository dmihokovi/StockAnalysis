import yfinance as yf

def collect_data():
    # Preuzimanje podataka o dionici Apple
    data = yf.download('AAPL', start='2022-01-01', end='2023-12-31')
    return data
