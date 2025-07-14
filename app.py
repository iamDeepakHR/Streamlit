import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import google.generativeai as genai

# ---- CONFIG ----
st.set_page_config(page_title="Stock Predictor", layout="wide")

# Paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ---- GEMINI SETUP (Optional) ----
# Replace with your Gemini API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-pro")

# ---- FUNCTIONS ----

def load_or_create_model(input_shape):
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(symbol):
    data = yf.Ticker(symbol).history(period="5y")
    if data.empty:
        st.error("No data found for this symbol.")
        return None, None

    prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(60, len(prices_scaled)):
        X.append(prices_scaled[i-60:i, 0])
        y.append(prices_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = load_or_create_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

def predict_next_30_days(model, scaler, symbol):
    data = yf.Ticker(symbol).history(period="3mo")
    prices = data["Close"].values.reshape(-1, 1)
    if len(prices) < 60:
        st.error("Not enough data for prediction.")
        return None

    scaled = scaler.transform(prices)
    last_60 = scaled[-60:].reshape((1, 60, 1))

    predictions_scaled = []
    for _ in range(30):
        pred = model.predict(last_60, verbose=0)
        predictions_scaled.append(pred[0, 0])
        pred_reshaped = pred.reshape(1, 1, 1)
        last_60 = np.concatenate((last_60[:, 1:, :], pred_reshaped), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    future_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, 31)]
    return pd.DataFrame({"Date": future_dates, "Predicted Close": predictions.flatten()})

def gemini_explain(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text

# ---- STREAMLIT UI ----

st.title("📈 Stock Analysis and Prediction")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")

col1, col2 = st.columns(2)

if st.button("Fetch Stock Data"):
    with st.spinner("Fetching data..."):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        if hist.empty:
            st.error("No data found.")
        else:
            col1.subheader("Actual Stock Prices (Last 3 Months)")
            col1.line_chart(hist["Close"])

            # Display stats
            info = ticker.info
            col2.write(f"**Company:** {info.get('shortName', 'N/A')}")
            col2.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
            col2.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
            col2.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")

if st.button("Train Model"):
    with st.spinner("Training LSTM model..."):
        model, scaler = train_model(symbol)
        if model:
            st.success("Model trained and saved!")

if st.button("Predict Next 30 Days"):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.warning("Please train the model first.")
    else:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        df_pred = predict_next_30_days(model, scaler, symbol)
        if df_pred is not None:
            st.subheader("Predicted Prices (Next 30 Days)")
            st.line_chart(df_pred.set_index("Date"))

# ---- GEMINI AI Assistant ----
if GEMINI_API_KEY:
    st.header("🤖 AI Stock Assistant (Gemini)")
    user_input = st.text_area("Ask about the stock (e.g., Explain AAPL trend)")
    if st.button("Ask Gemini"):
        with st.spinner("Gemini is thinking..."):
            prompt = f"Provide a financial analysis or explanation for the following stock: {symbol}. User query: {user_input}"
            response = gemini_explain(prompt)
            st.write(response)

