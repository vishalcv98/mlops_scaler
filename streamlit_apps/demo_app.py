import streamlit as st

st.title("This is a demo Streamlit app")


def add(a,b):
    return a+b

st.header("Addition of two numbers")
num1 = st.number_input("Enter first number", value=0)
num2 = st.number_input("Enter second number", value=0)

if st.button("Add"):
    st.write(f"The sum of {num1} and {num2} is: {add(num1, num2)}")
