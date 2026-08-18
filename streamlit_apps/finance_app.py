import yfinance as yf
import streamlit as st

st.title("Apple Stock Analysis App")

st.header("Select your company")

companies = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
comp = st.selectbox("Select a company:", companies)

data = yf.Ticker(comp)
df = data.history(start = "2019-01-01", end="2023-01-01")

tab1, tab2, tab3 = st.tabs(["Overview", "Price Chart analysis","Volume Analysis"])



with tab1:
    st.header("Company Overview")
    st.dataframe(df)

with tab2:
    st.header("Price Chart Analysis")
    st.line_chart(df['Close'])

with tab3:
    st.header("Volume Analysis")
    st.bar_chart(df['Volume'])



