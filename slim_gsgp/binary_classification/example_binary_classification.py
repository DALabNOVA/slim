# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Example: binary classification with SLIM_GSGP.

This follows the approach of Bakurov et al. (2022):
  - Training uses ``sigmoid_rmse`` so that raw outputs are squashed to [0, 1]
    before computing RMSE against binary labels.
  - Prediction uses ``binary_sign_transform``: negative raw output → class 0,
    non-negative raw output → class 1.

Reference:
    Bakurov, I., et al. (2022). General purpose optimization library (GPOL):
    A flexible and efficient multi-purpose optimization library in Python.
    Swarm and Evolutionary Computation, 68, 101028.
    https://doi.org/10.1016/j.swevo.2021.101028
"""

from sklearn.metrics import accuracy_score

from slim_gsgp.main_slim import slim
from slim_gsgp.datasets.data_loader import load_breast_cancer  # binary dataset
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.binary_classification import binary_sign_transform

# ---------------------------------------------------------------------------
# 1. Load a binary classification dataset
# ---------------------------------------------------------------------------
X, y = load_breast_cancer(X_y=True)

# ---------------------------------------------------------------------------
# 2. Split into train / validation / test
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# ---------------------------------------------------------------------------
# 3. Train with sigmoid_rmse
#    - fitness_function="sigmoid_rmse" wraps outputs with a sigmoid during
#      training so that RMSE is computed against values in [0, 1].
#    - sigmoid_scaling_factor controls the steepness of the sigmoid.
# ---------------------------------------------------------------------------
final_tree = slim(
    X_train=X_train,
    y_train=y_train,
    X_test=X_val,
    y_test=y_val,
    dataset_name="breast_cancer",
    slim_version="SLIM+ABS",
    pop_size=100,
    n_iter=100,
    ms_lower=0,
    ms_upper=1,
    p_inflate=0.5,
    fitness_function="sigmoid_rmse",
    sigmoid_scaling_factor=1.0,
)

# ---------------------------------------------------------------------------
# 4. Show the best individual
# ---------------------------------------------------------------------------
final_tree.print_tree_representation()

# ---------------------------------------------------------------------------
# 5. Predict on the test set and convert to binary labels
#    Raw outputs: negative → class 0, non-negative → class 1.
# ---------------------------------------------------------------------------
raw_predictions = final_tree.predict(X_test)
binary_predictions = binary_sign_transform(raw_predictions)

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test.numpy(), binary_predictions.numpy())
print(f"Test accuracy: {acc:.4f}")
