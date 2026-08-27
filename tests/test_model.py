import pickle

import numpy as np
import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "cars24-car-price-cleaned-new.csv"
MODEL_PATH = ROOT / "models" / "xgb_car_price_model.pkl"


SAMPLE_BATCH = np.array([
    [10000, 20, 5, 1, 0, 0],      # newer petrol car
    [120000, 19.7, 11, 1, 0, 0],  # old high-mileage petrol
    [50000, 22, 3, 0, 1, 0],      # diesel
    [30000, 25, 2, 0, 0, 1],      # electric
    [200000, 15, 15, 0, 1, 0],    # very old diesel
])


@pytest.fixture(scope="module")
def xgb_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_loads(xgb_model):
    assert xgb_model is not None



def test_predictions_are_positive(xgb_model):
    predictions = xgb_model.predict(SAMPLE_BATCH)
    assert (predictions > 0).all()


def test_model_predicts_batch(xgb_model):
    predictions = xgb_model.predict(SAMPLE_BATCH)
    assert predictions.shape == (len(SAMPLE_BATCH),)


def test_model_expects_six_features(xgb_model):
    assert xgb_model.n_features_in_ == 6


def test_older_car_is_cheaper(xgb_model):
    # km_driven, mileage, age, Petrol, Diesel, Electric
    newer_car = np.array([[30000, 20, 3, 1, 0, 0]])
    older_car = np.array([[30000, 20, 13, 1, 0, 0]])
    assert xgb_model.predict(older_car)[0] < xgb_model.predict(newer_car)[0]