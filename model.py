import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# آماده‌سازی داده
def prepare_data(data, n_steps=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['price']])
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i-n_steps:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# آموزش مدل و پیش‌بینی
def train_and_predict(df, n_days=10):
    n_steps = 10
    X, y, scaler = prepare_data(df, n_steps)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)

    # پیش‌بینی آینده
    last_seq = X[-1]
    preds = []
    for _ in range(n_days):
        pred = model.predict(last_seq.reshape(1, n_steps, 1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=n_days)
    return pd.Series(preds, index=future_dates)