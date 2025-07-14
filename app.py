# app.py ──────────────────────────────────────────────────────────
import os
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import google.generativeai as genai

# ── CONFIG ───────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Gemini API key (use Streamlit secrets or hard‑code for quick tests)
#GEMINI_API_KEY = st.secrets.get("gemini_api_key", "AIzaSyDGKBuSb5gi7l_OUq0p7tpdyj2S34_6TrM")      # best practice
GEMINI_API_KEY = "AIzaSyDGKBuSb5gi7l_OUq0p7tpdyj2S34_6TrM"                         # fallback

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None  # Gemini features will be hidden

# ── UTILS ────────────────────────────────────────────────────────
def load_or_create_model(input_shape):
    """Load saved model if it exists, otherwise build a fresh one."""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(symbol: str):
    """Download 5 years of daily data, train an LSTM, save model + scaler."""
    df = yf.download(symbol, period="5y", interval="1d", progress=False)

    if df.empty:
        st.error(f"No data found for symbol '{symbol}'.")
        return None, None

    prices = df["Close"].values.reshape(-1, 1)

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

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

def predict_next_30_days(model, scaler, symbol: str):
    """Generate a 30‑day rolling forecast."""
    df = yf.download(symbol, period="3mo", interval="1d", progress=False)
    if df.empty or len(df) < 60:
        st.error("Not enough recent data (need ≥ 60 rows).")
        return None

    prices = df["Close"].values.reshape(-1, 1)
    scaled = scaler.transform(prices)
    last_60 = scaled[-60:].reshape(1, 60, 1)

    preds_scaled = []
    for _ in range(30):
        pred = model.predict(last_60, verbose=0)
        preds_scaled.append(pred[0, 0])
        last_60 = np.concatenate((last_60[:, 1:, :], pred.reshape(1, 1, 1)), axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
    dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, 31)]
    return pd.DataFrame({"Date": dates, "Predicted Close": preds.flatten()})

def gemini_explain(prompt: str):
    """Call Gemini to answer a financial question."""
    if not gemini_model:
        return "Gemini API key not configured."
    return gemini_model.generate_content(prompt).text

# ── UI ───────────────────────────────────────────────────────────
st.title("📈 Stock Analysis and Prediction")

symbol_input = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
symbol = symbol_input.strip().upper()  # sanitise once

col1, col2 = st.columns(2)

# Fetch button ───────────────────────────────────────────────────
if st.button("Fetch Stock Data"):
    with st.spinner("Fetching data…"):
        hist = yf.download(symbol, period="3mo", interval="1d", progress=False)

        if hist.empty:
            st.error(f"No data found for symbol '{symbol}'.")
        else:
            col1.subheader("Actual Close Price (last 3 months)")
            col1.line_chart(hist["Close"])

            # Company fundamentals (wrap in try/except – sometimes fails)
            try:
                info = yf.Ticker(symbol).info
                col2.write(f"**Company:** {info.get('shortName', 'N/A')}")
                col2.write(f"**Market Cap:** {info.get('marketCap', 'N/A')}")
                col2.write(f"**PE Ratio:** {info.get('trailingPE', 'N/A')}")
                col2.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
            except Exception as e:
                col2.warning("Could not fetch company fundamentals.")
                col2.write(str(e))

# Train button ───────────────────────────────────────────────────
if st.button("Train Model"):
    with st.spinner("Training LSTM model (this may take a minute)…"):
        model, scaler = train_model(symbol)
        if model:
            st.success("Model trained ✔ and saved to disk!")

# Predict button ─────────────────────────────────────────────────
if st.button("Predict Next 30 Days"):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.warning("Please train the model first.")
    else:
        model   = load_model(MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        df_pred = predict_next_30_days(model, scaler, symbol)
        if df_pred is not None:
            st.subheader("Predicted Close Price (next 30 days)")
            st.line_chart(df_pred.set_index("Date"))

# Gemini assistant ───────────────────────────────────────────────
if gemini_model:
    st.header("🤖 AI Stock Assistant (Gemini)")
    user_query = st.text_area("Ask about the stock (e.g., ‘Explain AAPL trend over the last quarter’)")
    if st.button("Ask Gemini"):
        with st.spinner("Gemini is thinking…"):
            prompt = (
                f"Provide a concise financial analysis of {symbol}. "
                f"User question: {user_query}"
            )
            st.write(gemini_explain(prompt))
# ────────────────────────────────────────────────────────────────
