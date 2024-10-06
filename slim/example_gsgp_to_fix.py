from slim.main_gsgp import gsgp  # import the slim library
from slim.datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function
from slim.utils.utils import generate_random_uniform  # import the mutation step function

# Load the PPB dataset
X, y = load_ppb(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the Standard GSGP algorithm
final_tree = gsgp(X_train=X_train, y_train=y_train,
                  X_test=X_val, y_test=y_val, n_jobs=2,
                  dataset_name='ppb', pop_size=100, n_iter=100,
                  reconstruct=True, ms_lower=0, ms_upper=1)

# Get the prediction of the best individual on the test set
predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))

predictions = final_tree.predict(X_train)

print('train predictions {}...'.format(predictions[:5]))
print('train targets {}...'.format(y_train[:5]))
print('train rmse {}'.format(float(rmse(y_true=y_train, y_pred=predictions))))
