# test_app.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import precision_score, recall_score, f1_score

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
    model = load_model('./app/model.h5')
    assert model.evaluate(x_test, y_test)[1] > 0.5

def test_precision(test_data):
    x_test, y_test = test_data
    model = load_model('./app/model.h5')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    assert precision_score(y_test, y_pred, average='micro') > 0.5

def test_recall(test_data):
    x_test, y_test = test_data
    model = load_model('./app/model.h5')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    assert recall_score(y_test, y_pred, average='micro') > 0.5

def test_f1(test_data):
    x_test, y_test = test_data
    model = load_model('./app/model.h5')
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    assert f1_score(y_test, y_pred, average='micro') > 0.5
