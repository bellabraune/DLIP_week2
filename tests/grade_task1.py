import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import minimize # Python version of R's optim() function
from sklearn import datasets

from tasks import *

# unit test for first function


# Setting up common test parameters
np.random.seed(223)
w = np.random.normal(size=(6*4 + 4*7 + 7))  # vector with input weight values
X = np.random.normal(size=(6,10))  # matrix with 6 feature values for each of 10 simulated observations

# Test 1: Check output shape
def test_output_shape():
    output = my_mlp(w, X)
    assert output.shape == (1, 10), f"Expected shape (1, 10), but got {output.shape}"

# Test 2: Check output range for tanh
def test_output_range_tanh():
    output = my_mlp(w, X, sigma=np.tanh)
    assert np.all(output >= -1) and np.all(output <= 1), "Output is outside the expected tanh range of [-1, 1]"

# Test 3: Check output for a different activation function (e.g., sigmoid)
def test_output_sigmoid():
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    output = my_mlp(w, X, sigma=sigmoid)
    assert np.all(output >= 0) and np.all(output <= 1), "Output is outside the expected sigmoid range of [0, 1]"

# Test 4: Check output output for wrong handling of input
def test_output_value():
    output = np.round(my_mlp(w, X),7)
    expected_output = np.array([[-0.9985057 , -0.98895476,  0.98381408,  0.85878257, -0.98004358, -0.99413097, -0.99719557,  0.9290378 ,  0.66210054,  0.38851955]])
    expected_output = np.round(expected_output,7)
    assert np.max(abs(output - expected_output)) < 1e-9, "Failed on output values"
