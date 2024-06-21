# SLIM (Semantic Learning algorithm based on Inflate and deflate Mutation)

*gsgp_slim* is a Python library that implements the SLIM algorithm, which is a variant of the Geometric Semantic Genetic Programming (GSGP). This library includes functions for running standard Genetic Programming (GP), GSGP, and all developed versions of the SLIM algorithm. Users can specify the version of SLIM they wish to use and obtain results accordingly.

## Installation

To install the library, use the following command:
```sh
pip install gsgp_slim
```
Additionally, make sure to install all required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### Running GP 
To use the GP algorithm, you can use the following example:

```python
from slim.main_gp import gp  # import the slim library
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function

# Load the PPB dataset
X, y = load_ppb(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the GP algorithm
final_tree = gp(X_train=X_train, y_train=y_train,
                X_test=X_val, y_test=y_val,
                dataset_name='ppb', pop_size=100, n_iter=100)

# Show the best individual structure at the last generation
final_tree.print_tree_representation()

# Get the prediction of the best individual on the test set
predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))
```

### Running standard GSGP 
To use the GSGP algorithm, you can use the following example:

```python
from slim.main_gsgp import gsgp  # import the slim library
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
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
                  X_test=X_val, y_test=y_val,
                  dataset_name='ppb', pop_size=100, n_iter=100,
                  ms=generate_random_uniform(0, 1))

# Get the prediction of the best individual on the test set
predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))
```

### Running SLIM 
To use the SLIM GSGP algorithm, you can use the following example:

```python
from slim.main_slim import slim  # import the slim library
from datasets.data_loader import load_ppb  # import the loader for the dataset PPB
from slim.evaluators.fitness_functions import rmse  # import the rmse fitness metric
from slim.utils.utils import train_test_split  # import the train-test split function
from slim.utils.utils import generate_random_uniform  # import the mutation step function

# Load the PPB dataset
X, y = load_ppb(X_y=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)

# Split the test set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

# Apply the SLIM GSGP algorithm
final_tree = slim(X_train=X_train, y_train=y_train,
                  X_test=X_val, y_test=y_val,
                  dataset_name='ppb', slim_version='SLIM+SIG2', pop_size=100, n_iter=100,
                  ms=generate_random_uniform(0, 1), p_inflate=0.5)

# Show the best individual structure at the last generation
final_tree.print_tree_representation()

# Get the prediction of the best individual on the test set
predictions = final_tree.predict(X_test)

# Compute and print the RMSE on the test set
print(float(rmse(y_true=y_test, y_pred=predictions)))
```

## Arguments for the *gp*, *gsgp* and *slim* function

### Common arguments
* `X_train` : A torch tensor with the training input data *(default: None)*.
* `y_train` : A torch tensor with the training output data *(default: None)*.
* `X_test` : A torch tensor with the testing input data *(default: None)*.
* `y_test` : A torch tensor with the testing output data *(default: None)*. 
* `dataset_name` : A string specifying how the results will be logged *(default: None)*.
* `pop_size` : An integer specifying the population size *(default: 100)*.
* `n_iter` : An integer specifying the number of iterations *(default: 1000)*.
* `elitism` : A boolean specifying the presence of elitism *(default: True)*.
* `n_elites` : An integer specifying the number of elites *(default: 1)*.
* `init_depth` : An integer specifying the initial depth of the GP tree 
  * *default: 6* for gp and slim
  * *default: 8* for gsgp
* `log_path` : A string specifying where the results are going to be saved 
  * *default: 
    ``` os.path.join(os.getcwd(), "log", "gp.csv")```* for slim
  * *default: 
    ``` os.path.join(os.getcwd(), "log", "gsgp.csv")```* for slim
  * *default: 
    ``` os.path.join(os.getcwd(), "log", "slim.csv")```* for slim
* `seed`: An integer specifying the seed for randomness *(default: 1)*.

### Specific for *gp*

* `p_xo` : A float specifying the crossover probability *(default: 0.8)*.
* `max_depth` : An integer specifying the maximum depth of the GP tree *(default: 17)*.

### Specific for *gsgp*
* `p_xo` : A float specifying the crossover probability *(default: 0.0)*.
* * `ms`: A callable function to generate the mutation step *(default: generate_random_uniform(0, 1))*.

### Specific for *slim*
* `slim_version`: A string specifying the version of SLIM-GSGP to run *(default: "SLIM+SIG2")*.
* `ms`: A callable function to generate the mutation step *(default: generate_random_uniform(0, 1))*.
* `p_inflate`: A float specifying the probability to apply the inflate mutation *(default: 0.5)*.

