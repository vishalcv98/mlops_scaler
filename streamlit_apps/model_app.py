import pandas as pd
import streamlit as st
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_car_price_model.pkl"

# Must match the column order used in scripts/train_model.py
FEATURES = ['km_driven', 'mileage', 'age', 'Petrol', 'Diesel', 'Electric']

FUEL_ICONS = {"Petrol": "⛽", "Diesel": "🛢️", "Electric": "🔌"}

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# Card + hero styling. Cards carry their own background so they stay
# readable in both the light and dark Streamlit themes.
st.markdown(
    """
    <style>
      .hero {
          text-align: center;
          padding: 1.2rem 0 0.4rem 0;
      }
      .hero h1 {
          font-size: 2.6rem;
          font-weight: 800;
          margin: 0;
          background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
      }
      .hero p {
          opacity: 0.65;
          margin-top: 0.3rem;
          font-size: 0.95rem;
      }
      .price-card {
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
          border-radius: 18px;
          padding: 1.8rem 1rem;
          text-align: center;
          color: #ffffff;
          box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
          margin: 0.5rem 0 1.2rem 0;
      }
      .price-card .label {
          text-transform: uppercase;
          letter-spacing: 0.12em;
          font-size: 0.75rem;
          opacity: 0.85;
      }
      .price-card .value {
          font-size: 3rem;
          font-weight: 800;
          line-height: 1.15;
          margin: 0.3rem 0 0.2rem 0;
      }
      .price-card .sub {
          font-size: 0.85rem;
          opacity: 0.85;
      }
      .stButton > button {
          border-radius: 10px;
          font-weight: 600;
          height: 3rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# loading the model
@st.cache_resource #python decorator
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


st.markdown(
    """
    <div class="hero">
      <h1>🚗 Used Car Price Predictor</h1>
      <p>XGBoost model trained on 19,820 Cars24 listings</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"No model found at `{MODEL_PATH}`.\n\n"
        "Train one first:  `python scripts/train_model.py`"
    )
    st.stop()

xgb_model = load_model()

with st.sidebar:
    st.subheader("About")
    st.write(
        "Estimates the resale price of a used car from its usage, "
        "age and fuel type."
    )
    st.divider()
    st.caption("Model")
    st.write("XGBRegressor · 200 trees · depth 6")
    st.caption("Features")
    st.write(", ".join(FEATURES))

st.subheader("Car details")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Total distance on the odometer.",
        )
        age = st.slider("Age (years)", min_value=0, max_value=31, value=8)

    with col2:
        mileage = st.number_input(
            "Mileage (kmpl)",
            min_value=4.0,
            max_value=50.0,
            value=19.3,
            step=0.1,
            help="Fuel efficiency in kilometres per litre.",
        )
        fuel = st.radio(
            "Fuel type",
            ["Petrol", "Diesel", "Electric"],
            horizontal=True,
            format_func=lambda f: f"{FUEL_ICONS[f]} {f}",
        )

predict = st.button("Predict price", type="primary", use_container_width=True)

if predict:
    X = pd.DataFrame([{
        'km_driven': km_driven,
        'mileage': mileage,
        'age': age,
        'Petrol': int(fuel == "Petrol"),
        'Diesel': int(fuel == "Diesel"),
        'Electric': int(fuel == "Electric"),
    }])[FEATURES]

    price = float(xgb_model.predict(X)[0])

    st.markdown(
        f"""
        <div class="price-card">
          <div class="label">Estimated selling price</div>
          <div class="value">₹ {price:.2f}L</div>
          <div class="sub">≈ ₹ {price * 100000:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Age", f"{age} yr")
    c2.metric("Driven", f"{km_driven / 1000:,.0f}k km")
    c3.metric("Fuel", f"{FUEL_ICONS[fuel]} {fuel}")

    with st.expander("Model input"):
        st.dataframe(X, hide_index=True)

    if fuel == "Electric":
        st.warning(
            "The training data contains only 8 electric cars, "
            "so this prediction is unreliable."
        )
