from xgboost import XGBRegressor
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_car_price_model.pkl"

# load model
with open(MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

print(xgb_model.predict([[10000, 20, 5, 1, 0, 0]]))

