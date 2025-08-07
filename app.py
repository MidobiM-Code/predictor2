import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import fetch_realtime_price, load_and_prepare_data
from model import train_and_predict
from datetime import datetime

st.set_page_config(page_title="پیش‌بینی دلار با LSTM", layout="centered")
st.title("💵 پیش‌بینی قیمت دلار (نوبیتکس + LSTM)")

# دریافت قیمت لحظه‌ای
price = fetch_realtime_price()
if price:
    st.success(f"قیمت لحظه‌ای دلار (تتر نوبیتکس): {price:,.0f} تومان")
else:
    st.error("عدم دریافت قیمت لحظه‌ای از نوبیتکس")

# دریافت داده‌ها و آموزش مدل
days = st.slider("⏳ چند روز آینده رو پیش‌بینی کنم؟", 5, 30, 10)
data = load_and_prepare_data(price)
predicted = train_and_predict(data, n_days=days)

# نمایش نمودار
st.subheader("🔮 قیمت پیش‌بینی‌شده:")
fig, ax = plt.subplots()
ax.plot(predicted.index, predicted.values, label="پیش‌بینی", marker='o')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(fig)

# ذخیره پیش‌بینی
if st.button("💾 ذخیره در تاریخچه"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_save = pd.DataFrame({
        "timestamp": [now]*len(predicted),
        "date": predicted.index,
        "predicted_price": predicted.values
    })
    df_save.to_csv("history.csv", mode="a", index=False, header=False)
    st.success("پیش‌بینی ذخیره شد ✅")

# مشاهده تاریخچه
if st.checkbox("📜 نمایش تاریخچه پیش‌بینی‌ها"):
    try:
        history = pd.read_csv("history.csv", names=["timestamp", "date", "predicted_price"])
        st.dataframe(history.tail(30))
    except FileNotFoundError:
        st.warning("تاریخچه‌ای وجود ندارد.")