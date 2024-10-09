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
import os

import pandas
import pandas as pd
import torch

def load_pandas_df(df : pandas.DataFrame, X_y: bool = True):
    """

    Parameters
    ----------
    df : Pandas Dataframe
        The Pandas Dataframe that is to be used and turned into a torch.Tensor. Must contain the target variable
        in the last column if X_y is set to True.

    X_y : bool, optional
        Indicates if the data is to be returned as two objects of type torch.Tensor, otherwise as single Tensor.


    Returns
    -------

    tuple or torch.Tensor
        A tuple containing two torch.Tensors (X and the target variable, Y), if X_y is True or a single
        torch.Tensor if X_y is set to False

    """

    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return torch.from_numpy(df.values).float()


def load_resid_build_sale_price(X_y=True):
    """
    Loads and returns the RESIDNAME data set (regression). Taken from https://archive.ics.uci.edu/dataset/437/residential+building+data+set

    Parameters
    ----------
    X_y : bool, optional
        Indicates if the data is to be returned as two objects of type torch.Tensor, otherwise as single Tensor.

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
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "resid_build_sale_price.txt"), sep=" ",
        header=None
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df

def load_istanbul(X_y=True):
    """
    Loads and returns the Istanbul data set (regression). Taken from https://docs.1010data.com/MachineLearningExamples/IstanbulDataSet.html.

    Parameters
    ----------
    X_y : bool, optional
        Indicates if the data is to be returned as two objects of type torch.Tensor, otherwise as single Tensor.

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
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "istanbul.txt"), sep=" ",
        header=None
    )
    if X_y:
        return (
            torch.from_numpy(df.values[:, :-1]).float(),
            torch.from_numpy(df.values[:, -1]).float(),
        )
    else:
        return df



#   The following functions were Adapted from the GPOL library.

def load_airfoil(X_y=True):
    """Loads and returns the Airfoil Self-Noise data set (regression)

    NASA data set, obtained from a series of aerodynamic and acoustic
    tests of two and three-dimensional airfoil blade sections conducted
    in an anechoic wind tunnel.
    Downloaded from the UCI ML Repository.
    The file is located in slim/datasets/data/airfoil.txt

    Basic information:
    - Number of data instances: 1503;
    - Number of input features: 5;
    - Target's range: [103.38-140.987].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_boston(X_y=True):
    """Loads and returns the Boston Housing data set (regression)

    This dataset contains information collected by the U.S. Census
    Service concerning housing in the area of Boston Massachusetts.
    Downloaded from the StatLib archive.
    The file is located in /slim/datasets/data/boston.txt

    Basic information:
    - Number of data instances: 506;
    - Number of input features: 13;
    - Target's range: [5, 50].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_breast_cancer(X_y=True):
    """Loads and returns the breast cancer data set (classification)

    Breast Cancer Wisconsin (Diagnostic) dataset.
    Downloaded from the StatLib archive.
    The file is located in /slim/datasets/data/boston.txt

    Basic information:
    - Number of data instances: 569;
    - Number of input features: 30;
    - Target's values: {0: "benign", 1: "malign"}.

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_concrete_slump(X_y=True):
    """Loads and returns the Concrete Slump data set (regression)

    Concrete is a highly complex material. The slump flow of concrete
    is not only determined by the water content, but that is also
    influenced by other concrete ingredients.
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/concrete_slump.txt

    Basic information:
    - Number of data instances: 103;
    - Number of input features: 7;
    - Target's range: [0, 29].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_concrete_strength(X_y=True):
    """Loads and returns the Concrete Strength data set (regression)

    Concrete is the most important material in civil engineering. The
    concrete compressive strength is a highly nonlinear function of
    age and ingredients.
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/concrete_strength.txt

    Basic information:
    - Number of data instances: 1005;
    - Number of input features: 8;
    - Target's range: [2.331807832, 82.5992248].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_diabetes(X_y=True):
    """Loads and returns the Diabetes data set(regression)

    The file is located in /slim/datasets/data/diabetes.txt

    Basic information:
    - Number of data instances: 442;
    - Number of input features: 10;
    - Target's range: [25, 346].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_efficiency_heating(X_y=True):
    """Loads and returns the Heating Efficiency data set(regression)

    The data set regards heating load assessment of buildings (that is,
    energy efficiency) as a function of building parameters.
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/efficiency_heating.txt

    Basic information:
    - Number of data instances: 768;
    - Number of input features: 8;
    - Target's range: [6.01, 43.1].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_efficiency_cooling(X_y=True):
    """Loads and returns the Cooling Efficiency data set(regression)

    The data set regards cooling load assessment of buildings (that is,
    energy efficiency) as a function of building parameters.
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/efficiency_cooling.txt

    Basic information:
    - Number of data instances: 768;
    - Number of input features: 8;
    - Target's range: [10.9, 48.03].

    Parameters
    ----------
    X_y : bool, optional
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    X_y : bool, optional
        The input data (X) and the target of the prediction (y). The
        latter is extracted from the data set as the last column.
    df : pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_forest_fires(X_y=True):
    """Loads and returns the Forest Fires data set (regression)

    The data set regards the prediction of the burned area of forest
    fires, in the northeast region of Portugal, by using meteorological
    and other data.
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/forest_fires.txt

    Basic information:
    - Number of data instances: 513;
    - Number of input features: 43;
    - Target's range: [0.0, 6.995619625423205].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_parkinson_updrs(X_y=True):
    """Loads and returns the Parkinsons Telemonitoring data set (regression)

    The data set was created by A. Tsanas and M. Little of the Oxford's
    university in collaboration with 10 medical centers in the US and
    Intel Corporation who developed the telemonitoring device to record
    the speech signals. The original study used a range of linear and
    nonlinear regression methods to predict the clinician's Parkinson's
    disease symptom score on the UPDRS scale (total UPDRS used here).
    Downloaded from the UCI ML Repository.
    The file is located in /slim/datasets/data/parkinson_total_UPDRS.txt

    Basic information:
    - Number of data instances: 5875;
    - Number of input features: 19;
    - Target's range: [7.0, 54.992].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_ld50(X_y=True):
    """Loads and returns the LD50 data set(regression)

    The data set consists in predicting the median amount of compound
    required to kill 50% of the test organisms (cavies), also called
    the lethal dose or LD50. For more details, consult the publication
    entitled as "Genetic programming for computational pharmacokinetics
    in drug discovery and development" by F. Archetti et al. (2007).
    The file is located in /slim/datasets/data/ld50.txt

    Basic information:
    - Number of data instances: 234;
    - Number of input features: 626;
    - Target's range: [0.25, 8900.0].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_ppb(X_y=True):
    """Loads and returns the PPB data set(regression)

    The data set consists in predicting the percentage of the initial
    drug dose which binds plasma proteins (also known as the plasma
    protein binding level). For more details, consult the publication
    entitled as "Genetic programming for computational pharmacokinetics
    in drug discovery and development" by F. Archetti et al. (2007).
    The file is located in /slim/datasets/data/ppb.txt

    Basic information:
    - Number of data instances: 131;
    - Number of input features: 626;
    - Target's range: [0.5, 100.0]

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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


def load_bioav(X_y=True):
    """Loads and returns the Oral Bioavailability data set (regression)

    The data set consists in predicting the value of the percentage of
    the initial orally submitted drug dose that effectively reaches the
    systemic blood circulation after being filtered by the liver, as a
    function of drug's molecular structure. For more details, consult
    the publication entitled as "Genetic programming for computational
    pharmacokinetics in drug discovery and development" by F. Archetti
    et al. (2007).
    The file is located in slim/datasets/data/bioavailability.txt

    Basic information:
    - Number of data instances: 358;
    - Number of input features: 241;
    - Target's range: [0.4, 100.0].

    Parameters
    ----------
    X_y : bool, optional
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

    References
    ----------
    Bakurov, I., Buzzelli, M., Castelli, M., Vanneschi, L., & Schettini, R. (2021). General purpose optimization
    library (GPOL): a flexible and efficient multi-purpose optimization library in Python. Applied Sciences, 11(11),
    4774.
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
