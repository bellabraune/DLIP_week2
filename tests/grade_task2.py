import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize # Python version of R's optim() function
from sklearn import datasets

from tasks import *

# unit test for second function

# Setting up common test parameters
np.random.seed(223)
w = np.random.normal(size=(6*4 + 4*7 + 7))  # vector with input weight values
X = np.random.normal(size=(6, 10))  # matrix with 6 feature values for each of 10 simulated observations
y = np.array([1 if i < X.shape[1] // 2 else -1 for i in range(X.shape[1])])  # target labels

# Test 1: Check output type
def test_mse_output_type():
    mse = MSE_func(w, X, y)
    assert isinstance(mse, (float, np.floating)), f"Expected output type float, but got {type(mse)}"

# Test 2: Check non-negative MSE
def test_mse_non_negative():
    mse = MSE_func(w, X, y)
    assert mse >= 0, f"Expected non-negative MSE, but got {mse}"

# Test 3: Check MSE with uniform weights and features
def test_mse_uniform_input():
    w_uniform = np.ones(shape=(6*4 + 4*7 + 7))
    X_uniform = np.ones(shape=(6, 10))
    mse = MSE_func(w_uniform, X_uniform, y)
    assert abs(mse-20) < 1e-4, f"Wrong value: {mse}"
    w_rev = w
    X_uniform = X/20
    mse = MSE_func(w_rev, X_uniform, y)
    assert abs(mse-12.220975600119619) < 1e-8, f"Wrong value (2): {mse}"
