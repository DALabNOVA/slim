import os

import pandas as pd
import torch



def load_merged_data(dataset, X_y=False):

    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "merged_data",
            f"{dataset}_merged.txt",
        ),
        sep=" ",
        header=None,
    )

    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_dummy_test(boo=True):
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "TEST_1_myToySumDataset.txt",
        ),
        sep=" ",
        header=None,
    )

    return (
        torch.from_numpy(df.values[:, :-2]).float(),
        torch.from_numpy(df.values[:, -2]).float(),
    )


def load_dummy_train(boo=True):
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "TRAINING_1_myToySumDataset.txt",
        ),
        sep=" ",
        header=None,
    )

    return (
        torch.from_numpy(df.values[:, :-2]).float(),
        torch.from_numpy(df.values[:, -2]).float(),
    )


def load_preloaded(dataset_name, seed = 1, training=True, X_y=False):

    filename = (
        f"TRAINING_{seed}_{dataset_name.upper()}.txt"
        if training
        else f"TEST_{seed}_{dataset_name.upper()}.txt"
    )

    # dropping the last column as it only contains NaNs due to spacing as
    # separator
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "pre_loaded_data", filename
        ),
        sep=" ",
        header=None,
    ).iloc[:, :-1]

    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


"""

Taken from GPOL.

"""
def load_airfoil(X_y=False):
    """Loads and returns the Airfoil Self-Noise data set (regression)

    NASA data set, obtained from a series of aerodynamic and acoustic
    tests of two and three-dimensional airfoil blade sections conducted
    in an anechoic wind tunnel.
    Downloaded from the UCI ML Repository.
    The file is located in gpol/utils/data/airfoil.txt

    Basic information:
    - Number of data instances: 1503;
    - Number of input features: 5;
    - Target's range: [103.38-140.987].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "airfoil.txt")
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_boston(X_y=False):
    """Loads and returns the Boston Housing data set (regression)

    This dataset contains information collected by the U.S. Census
    Service concerning housing in the area of Boston Massachusetts.
    Downloaded from the StatLib archive.
    The file is located in /gpol/utils/data/boston.txt

    Basic information:
    - Number of data instances: 506;
    - Number of input features: 13;
    - Target's range: [5, 50].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    X, y : torch.Tensor, torch.Tensor
        The input data (X) and the target of the prediction (y). The
        latter is extracted from the data set as the last column.
    df : pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "boston.txt")
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_breast_cancer(X_y=False):
    """Loads and returns the breast cancer data set (classification)

    Breast Cancer Wisconsin (Diagnostic) dataset.
    Downloaded from the StatLib archive.
    The file is located in /gpol/utils/data/boston.txt

    Basic information:
    - Number of data instances: 569;
    - Number of input features: 30;
    - Target's values: {0: "benign", 1: "malign"}.

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    X, y : torch.Tensor, torch.Tensor
        The input data (X) and the target of the prediction (y). The
        latter is extracted from the data set as the last column.
    df : pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "breast_cancer.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_concrete_slump(X_y=False):
    """Loads and returns the Concrete Slump data set (regression)

    Concrete is a highly complex material. The slump flow of concrete
    is not only determined by the water content, but that is also
    influenced by other concrete ingredients.
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/concrete_slump.txt

    Basic information:
    - Number of data instances: 103;
    - Number of input features: 7;
    - Target's range: [0, 29].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "concrete_slump.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_concrete_strength(X_y=False):
    """Loads and returns the Concrete Strength data set (regression)

    Concrete is the most important material in civil engineering. The
    concrete compressive strength is a highly nonlinear function of
    age and ingredients.
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/concrete_strength.txt

    Basic information:
    - Number of data instances: 1005;
    - Number of input features: 8;
    - Target's range: [2.331807832, 82.5992248].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "concrete_strength.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_diabetes(X_y=False):
    """Loads and returns the Diabetes data set(regression)

    The file is located in /gpol/utils/data/diabetes.txt

    Basic information:
    - Number of data instances: 442;
    - Number of input features: 10;
    - Target's range: [25, 346].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    X, y : torch.Tensor, torch.Tensor
        The input data (X) and the target of the prediction (y). The
        latter is extracted from the data set as the last column.
    df : pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "diabetes.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_efficiency_heating(X_y=False):
    """Loads and returns the Heating Efficiency data set(regression)

    The data set regards heating load assessment of buildings (that is,
    energy efficiency) as a function of building parameters.
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/efficiency_heating.txt

    Basic information:
    - Number of data instances: 768;
    - Number of input features: 8;
    - Target's range: [6.01, 43.1].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "efficiency_heating.txt",
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_efficiency_cooling(X_y=False):
    """Loads and returns the Cooling Efficiency data set(regression)

    The data set regards cooling load assessment of buildings (that is,
    energy efficiency) as a function of building parameters.
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/efficiency_cooling.txt

    Basic information:
    - Number of data instances: 768;
    - Number of input features: 8;
    - Target's range: [10.9, 48.03].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "efficiency_cooling.txt",
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_forest_fires(X_y=False):
    """Loads and returns the Forest Fires data set (regression)

    The data set regards the prediction of the burned area of forest
    fires, in the northeast region of Portugal, by using meteorological
    and other data.
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/forest_fires.txt

    Basic information:
    - Number of data instances: 513;
    - Number of input features: 43;
    - Target's range: [0.0, 6.995619625423205].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "forest_fires.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_parkinson_updrs(X_y=False):
    """Loads and returns the Parkinsons Telemonitoring data set (regression)

    The data set was created by A. Tsanas and M. Little of the Oxford's
    university in collaboration with 10 medical centers in the US and
    Intel Corporation who developed the telemonitoring device to record
    the speech signals. The original study used a range of linear and
    nonlinear regression methods to predict the clinician's Parkinson's
    disease symptom score on the UPDRS scale (total UPDRS used here).
    Downloaded from the UCI ML Repository.
    The file is located in /gpol/utils/data/parkinson_total_UPDRS.txt

    Basic information:
    - Number of data instances: 5875;
    - Number of input features: 19;
    - Target's range: [7.0, 54.992].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data",
            "parkinson_total_UPDRS.txt",
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_ld50(X_y=False):
    """Loads and returns the LD50 data set(regression)

    The data set consists in predicting the median amount of compound
    required to kill 50% of the test organisms (cavies), also called
    the lethal dose or LD50. For more details, consult the publication
    entitled as "Genetic programming for computational pharmacokinetics
    in drug discovery and development" by F. Archetti et al. (2007).
    The file is located in /gpol/utils/data/ld50.txt

    Basic information:
    - Number of data instances: 234;
    - Number of input features: 626;
    - Target's range: [0.25, 8900.0].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "ld50.txt")
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_ppb(X_y=False):
    """Loads and returns the PPB data set(regression)

    The data set consists in predicting the percentage of the initial
    drug dose which binds plasma proteins (also known as the plasma
    protein binding level). For more details, consult the publication
    entitled as "Genetic programming for computational pharmacokinetics
    in drug discovery and development" by F. Archetti et al. (2007).
    The file is located in /gpol/utils/data/ppb.txt

    Basic information:
    - Number of data instances: 131;
    - Number of input features: 626;
    - Target's range: [0.5, 100.0]

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "ppb.txt")
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df


def load_bioav(X_y=False):
    """Loads and returns the Oral Bioavailability data set (regression)

    The data set consists in predicting the value of the percentage of
    the initial orally submitted drug dose that effectively reaches the
    systemic blood circulation after being filtered by the liver, as a
    function of drug's molecular structure. For more details, consult
    the publication entitled as "Genetic programming for computational
    pharmacokinetics in drug discovery and development" by F. Archetti
    et al. (2007).
    The file is located in gpol/utils/data/bioavailability.txt

    Basic information:
    - Number of data instances: 358;
    - Number of input features: 241;
    - Target's range: [0.4, 100.0].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "bioavailability.txt"
        )
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df
