from slim.main_gp import gp  # import the slim library
from slim.datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function

import pandas as pd


# Load the PPB dataset
X, y = load_ppb(X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

final_tree = gp(X_train=X_train, y_train=y_train,
                X_test=X_val, y_test=y_val,
                dataset_name='ppb', pop_size=100, n_iter=100, )

predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))

