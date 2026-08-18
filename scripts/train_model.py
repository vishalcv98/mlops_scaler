import pandas as pd
from xgboost import XGBRegressor
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_car_price_model.pkl"

cars_df = pd.read_csv(ROOT / "data" / "cars24-car-price-cleaned-new.csv")

X = cars_df[['km_driven','mileage','age','Petrol','Diesel','Electric']]
y = cars_df['selling_price']   # better as 1D array

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=6
)

# Fit model
xgb_model.fit(X, y)

# Save model
# wb - write binary
MODEL_PATH.parent.mkdir(exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(xgb_model, f)


