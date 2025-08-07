import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# دریافت قیمت تتر از نوبیتکس
def fetch_realtime_price():
    try:
        res = requests.get("https://api.nobitex.ir/market/stats")
        price = float(res.json()['stats']['usdt-rls']['latest'])
        return price
    except:
        return None

# ساخت داده مصنوعی بر اساس قیمت لحظه‌ای
def load_and_prepare_data(current_price, days=180):
    dates = pd.date_range(end=datetime.today(), periods=days)
    prices = np.linspace(current_price * 0.6, current_price * 1.05, days) + np.random.normal(0, 5000, days)
    df = pd.DataFrame({"date": dates, "price": prices})
    return df