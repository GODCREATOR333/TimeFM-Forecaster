import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests

MODEL_NAME = "google/timefm-base"  # Placeholder name

def download_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

def calculate_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['Signal_Line'], _ = calculate_macd(df['Close'])
    return df

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def get_macro_indicators(start_date, end_date):
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)['Close']
    interest_rates = yf.download("^TNX", start=start_date, end=end_date)['Close']
    bond_yields = yf.download("^TYX", start=start_date, end=end_date)['Close']
    inflation = pd.read_csv('inflation_data.csv', index_col='Date', parse_dates=True)['Inflation']
    vix = yf.download("^VIX", start=start_date, end=end_date)['Close']
    
    return pd.DataFrame({
        'USD_Index': usd_index,
        'Interest_Rates': interest_rates,
        'Bond_Yields': bond_yields,
        'Inflation': inflation,
        'VIX': vix
    })

def get_index_specific_indicators(index_ticker, start_date, end_date):
    index_data = yf.Ticker(index_ticker)
    
    # Fetch quarterly financial data
    financials = index_data.quarterly_financials
    
    # Calculate P/E ratio
    pe_ratio = index_data.info.get('trailingPE', np.nan)
    
    # Calculate Earnings Growth YoY
    if not financials.empty and 'Net Income' in financials.index:
        net_income = financials.loc['Net Income']
        earnings_growth = ((net_income.iloc[0] - net_income.iloc[4]) / net_income.iloc[4]) * 100 if len(net_income) >= 5 else np.nan
    else:
        earnings_growth = np.nan
    
    # Get index-specific data (example for S&P 500)
    if index_ticker == "^GSPC":
        sp500_pe = yf.Ticker("^SP500-PE-RATIO").history(start=start_date, end=end_date)['Close']
        sp500_div_yield = yf.Ticker("^SP500-DIV-YLD").history(start=start_date, end=end_date)['Close']
        return pd.DataFrame({
            'PE_Ratio': sp500_pe,
            'Dividend_Yield': sp500_div_yield,
            'Earnings_Growth_YoY': pd.Series([earnings_growth] * len(sp500_pe), index=sp500_pe.index)
        })
    
    # For other indices, return available data
    return pd.DataFrame({
        'PE_Ratio': pd.Series([pe_ratio] * len(index_data.history(start=start_date, end=end_date)), index=index_data.history(start=start_date, end=end_date).index),
        'Earnings_Growth_YoY': pd.Series([earnings_growth] * len(index_data.history(start=start_date, end=end_date)), index=index_data.history(start=start_date, end=end_date).index)
    })

def prepare_features(index_data, macro_data, index_specific_data):
    features = pd.concat([index_data, macro_data, index_specific_data], axis=1).dropna()
    features['Returns'] = features['Close'].pct_change()
    return features.dropna()

def prepare_timefm_input(features, sequence_length=30):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    input_sequences = []
    for i in range(len(scaled_features) - sequence_length):
        input_sequences.append(scaled_features[i:i+sequence_length])
    
    return np.array(input_sequences), scaler

def load_timefm_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def predict_with_timefm(model, tokenizer, input_sequence):
    input_ids = tokenizer.encode(str(input_sequence.tolist()), return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=input_ids.shape[1] + 1, num_return_sequences=1)
    
    prediction = tokenizer.decode(output[0])
    return float(prediction.strip())

def categorize_prediction(prediction):
    if prediction > 0.02:
        return "Highly Positive (10%)"
    elif prediction > 0.005:
        return "Positive (50%)"
    elif prediction > -0.005:
        return "Neutral (5%)"
    elif prediction > -0.02:
        return "Slightly Negative (10%)"
    else:
        return "Negative (25%)"

def main():
    # Set parameters
    index_ticker = "^GSPC"  # S&P 500
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
    
    # Download and prepare data
    index_data = download_data(index_ticker, start_date, end_date)
    index_data = calculate_technical_indicators(index_data)
    macro_data = get_macro_indicators(start_date, end_date)
    index_specific_data = get_index_specific_indicators(index_ticker, start_date, end_date)
    features = prepare_features(index_data, macro_data, index_specific_data)
    
    # Prepare input for TimeFM
    input_sequences, scaler = prepare_timefm_input(features)
    
    # Load TimeFM model
    tokenizer, model = load_timefm_model()
    
    # Make prediction for the next day
    latest_sequence = input_sequences[-1]
    prediction = predict_with_timefm(model, tokenizer, latest_sequence)
    
    # Inverse transform the prediction
    original_scale_prediction = scaler.inverse_transform([[prediction]])[0][0]
    
    category = categorize_prediction(original_scale_prediction)
    
    print(f"Prediction for next day: {category}")
    print(f"Predicted return: {original_scale_prediction:.2%}")
    
    # Visualize recent price movement and prediction
    plt.figure(figsize=(12, 6))
    plt.plot(index_data.index[-30:], index_data['Close'][-30:], label='Closing Price')
    plt.title(f"{index_ticker} - Recent Price Movement and Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.annotate(f"Prediction: {category}", xy=(0.05, 0.95), xycoords='axes fraction')
    plt.show()

if __name__ == "__main__":
    main()