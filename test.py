# test_app.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pytest

@pytest.fixture(scope="module")
def test_data():
    test_data = pd.read_csv('./Dataset/sign_mnist_test/sign_mnist_test.csv')
    y_test = test_data['label']
    x_test = test_data.drop('label', axis=1)
    x_test = np.array(x_test.values)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_test, y_test

def test_accuracy(test_data):
    x_test, y_test = test_data
    model = load_model('./app/model.pkl')
    assert model.evaluate(x_test, y_test)[1] > 0.7
