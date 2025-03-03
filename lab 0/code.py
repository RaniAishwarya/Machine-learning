import pandas as pd

# Create a DataFrame directly from a dictionary
data = {
    'USN': ['1MS23IS001', '1MS23IS002', '1MS23IS003', '1MS23IS004', '1MS23IS005'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Marks': [85, 92, 78, 88, 95]
}

df = pd.DataFrame(data)

print("DataFrame with initialized values:")
print(df)



from sklearn.datasets import load_diabete
import pandas as pd

diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

df['target'] = diabetes.target

print("Sample data:")

print(df.head())




import pandas as pd

# Load data from a CSV file
file_path = '/content/sales_data_sample.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, encoding='latin1') # Try 'latin1' encoding

print("Sample data:")
print(df.head())




import pandas as pd

# Load data from a CSV file
file_path = '/content/Dataset of Diabetes .csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

print("Sample data:")
print(df.head())






import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the tickers for the banks
tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS"]

# Download historical data with new start and end dates
data = yf.download(tickers, start="2024-01-01", end="2024-12-30", group_by='ticker')

# Print some information about the data
print("First 5 rows of the dataset:")
print(data.head())






print("\nShape of the dataset:")
print(data.shape)
print("\nColumn names:")
print(data.columns)





for ticker in tickers:
    bank_data = data[ticker]
    print(f"\nSummary statistics for {ticker}:")
    print(bank_data.describe())
    bank_data['Daily Return'] = bank_data['Close'].pct_change()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    bank_data['Close'].plot(title=f"{ticker} - Closing Price")
    plt.subplot(2, 1, 2)
    bank_data['Daily Return'].plot(title=f"{ticker} - Daily Returns", color='orange')
    plt.tight_layout()
    plt.show()

    bank_data.to_csv(f'{ticker}_stock_data.csv')
    print(f"\n{ticker} stock data saved to '{ticker}_stock_data.csv'.")





