import yfinance as yf
import pandas as pd
import datetime

def download_stock_data(ticker_symbol, start_date, end_date):
    # Download historical data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Calculate missing features
    stock_data['Change'] = stock_data['Close'] - stock_data['Adj Close']
    stock_data['Percentage Change'] = (stock_data['Close'] - stock_data['Adj Close']) / stock_data['Adj Close'] * 100
    stock_data['Amount'] = stock_data['Close'] * stock_data['Volume']

    # Reorder columns to match the desired format
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                             'Change', 'Percentage Change', 'Amount']]
    return stock_data

def export_to_csv(stock_data, filename):
    # Export data to a CSV file
    stock_data.to_csv(filename)
    print(f"Modified stock data has been exported to {filename}")

if __name__ == "__main__":
    # Define the ticker symbol and date range for the historical data
    ticker_symbol = input("Enter the ticker symbol: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')  # To get data until yesterday

    # Download stock data
    stock_data = download_stock_data(ticker_symbol, start_date, end_date)

    # Export data to a CSV file
    csv_filename = f"{ticker_symbol.lower()}_stock_data.csv"
    export_to_csv(stock_data, csv_filename)
