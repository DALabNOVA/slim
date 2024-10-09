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
import csv
import os.path
from copy import copy
from uuid import UUID

import pandas as pd


def log_settings(path: str, settings_dict: list, unique_run_id: UUID) -> None:
    """
    Log the settings to a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    settings_dict : list
        Dictionary of settings.
    unique_run_id : UUID
        Unique identifier for the run.

    Returns
    -------
    None
    """
    settings_dict = merge_settings(*settings_dict)
    del settings_dict["TERMINALS"]

    infos = [unique_run_id, settings_dict]

    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(infos)


def merge_settings(sd1: dict, sd2: dict, sd3: dict, sd4: dict) -> dict:
    """
    Merge multiple settings dictionaries into one.

    Parameters
    ----------
    sd1 : dict
        First settings dictionary.
    sd2 : dict
        Second settings dictionary.
    sd3 : dict
        Third settings dictionary.
    sd4 : dict
        Fourth settings dictionary.

    Returns
    -------
    dict
        Merged settings dictionary.
    """
    return {**sd1, **sd2, **sd3, **sd4}


def logger(
    path: str,
    generation: int,
    elite_fit: float,
    timing: float,
    nodes: int,
    additional_infos: list = None,
    run_info: list = None,
    seed: int = 0,
) -> None:
    """
    Logs information into a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    generation : int
        Current generation number.
    elite_fit : float
        Elite's validation fitness value.
    timing : float
        Time taken for the process.
    nodes : int
        Count of nodes in the population.
    additional_infos : list, optional
        Population's test fitness value(s) and diversity measurements. Defaults to None.
    run_info : list, optional
        Information about the run. Defaults to None.
    seed : int, optional
        The seed used in random, numpy, and torch libraries. Defaults to 0.

    Returns
    -------
    None
    """
    if not os.path.isdir(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        infos = copy(run_info) if run_info is not None else []
        infos.extend([seed, generation, float(elite_fit), timing, nodes])

        if additional_infos is not None:
            try:
                additional_infos[0] = float(additional_infos[0])
            except:
                additional_infos[0] = "None"
            infos.extend(additional_infos)

        writer.writerow(infos)


def drop_experiment_from_logger(experiment_id: str or int, log_path: str) -> None:
    """
    Remove an experiment from the logger CSV file. If the given experiment_id is -1, the last saved experiment is removed.

    Parameters
    ----------
    experiment_id : str or int
        The experiment ID to be removed. If -1, the most recent experiment is removed.
    log_path : str
        Path to the file containing the logging information.

    Returns
    -------
    None
    """
    logger_data = pd.read_csv(log_path)

    # If we choose to remove the last stored experiment
    if experiment_id == -1:
        # Find the experiment id of the last row in the CSV file
        experiment_id = logger_data.iloc[-1, 1]

    # Exclude the logger data with the chosen id
    to_keep = logger_data[logger_data.iloc[:, 1] != experiment_id]
    # Save the new excluded dataset
    to_keep.to_csv(log_path, index=False, header=None)
