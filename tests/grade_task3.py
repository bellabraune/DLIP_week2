import numpy as np
from tasks import *

# unit test for third function

# Setting up common test parameters
beta = np.array([-0.1, 1.1])

# Test 1: Check output shape
def test_dR_output_shape():
    x = np.arange(-3, 3)
    y = np.arange(-3, 3) + 5
    d = dR(beta, x, y)
    assert d.shape == (2,), f"Expected shape (2,), but got {d.shape}"

# Test 2: Check output with all zeros for x and y
def test_dR_zeros_input():
    x = np.zeros(6)
    y = np.zeros(6)
    d = dR(beta, x, y)
    expected = np.array([2*beta[0], 0.0])  # Since x and y are zeros, derivatives should be zero
    assert np.allclose(d, expected), f"Expected {expected}, but got {d}"

# Test 3: Check dR with non-linear x values (squared values)
def test_dR_non_linear_x():
    x = np.arange(-3, 3)**2
    y = np.arange(-3, 3) + 5
    d = dR(beta, x, y)
    assert d.shape == (2,), f"Expected shape (2,), but got {d}"
    assert abs(d[0]+2.3-0.2/3) < 1e-7, f"Wrong value error: {d[0]}"
