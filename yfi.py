import sqlite3
import yfinance as yf
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define stock ticker and update interval
ticker_symbol = "AAPL"  # Example: Apple Inc.
update_interval = 60  # Fetch data every 60 seconds

def fetch_stock_data(ticker, period="5y", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
    return df

def save_to_db(df, db_file="db.sql"):
    conn = sqlite3.connect(db_file)
    df.to_sql("stock_data", conn, if_exists="replace", index=False)
    conn.close()

def load_from_db(db_file="db.sql"):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql("SELECT * FROM stock_data", conn)
    conn.close()
    return df

def train_and_predict(df):
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = (df["date"] - df["date"].min()).dt.days
    
    train_df = df[df["date"] < "2023-01-01"]
    test_df = df[df["date"] >= "2023-01-01"]
    
    X_train = train_df[["day"]]
    y_train = train_df["price"]
    X_test = test_df[["day"]]
    y_test = test_df["price"]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices', alpha=0.5)
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
    plt.xlabel("Days since start")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.show()
    
    return model.score(X_test_scaled, y_test)

# Fetch and save data
df = fetch_stock_data(ticker_symbol)
save_to_db(df)
df_loaded = load_from_db()
accuracy = train_and_predict(df_loaded)
print(f"Model Accuracy: {accuracy:.2f}")

# Live Data Update
def update_live_data():
    while True:
        print("Fetching latest stock data...")
        df_new = fetch_stock_data(ticker_symbol, period="7d", interval="1h")
        save_to_db(df_new)
        print("Database updated.")
        time.sleep(update_interval)  # Wait before next update

# Uncomment below line to enable live updates
# update_live_data()

