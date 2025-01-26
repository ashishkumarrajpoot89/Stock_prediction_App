# Stock Price Prediction using LSTM

## About the Project
This project demonstrates a machine learning approach to predict stock prices using Long Short-Term Memory (LSTM) neural networks. By fetching historical stock data from Yahoo Finance, the project preprocesses the data, trains an LSTM model, and enables predictions of future stock prices based on historical trends.

---

## Features
- Fetch historical stock price data using the **yfinance** library.
- Preprocess stock data with MinMax scaling for better training performance.
- Train a Long Short-Term Memory (LSTM) neural network for stock price prediction.
- Use the past 60 days of stock prices to predict the next day's closing price.

---

## How It Works
1. **Data Collection**:
   - Stock data is fetched from Yahoo Finance using the `yfinance` library.
   - Only the "Close" price is used for prediction.

2. **Data Preprocessing**:
   - Scale the stock prices between 0 and 1 using `MinMaxScaler`.
   - Prepare training data by creating sequences of the past 60 days' prices as input and the next day's price as the target.

3. **Model Training**:
   - Build an LSTM-based Sequential model using TensorFlow/Keras.
   - Train the model for 10 epochs with a batch size of 32.

4. **Prediction**:
   - Use the trained model to predict future stock prices.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `NumPy`: For numerical operations.
  - `Pandas`: For data manipulation.
  - `yfinance`: For fetching stock data.
  - `TensorFlow/Keras`: For building and training the LSTM model.
  - `scikit-learn`: For data preprocessing (MinMaxScaler).

---
