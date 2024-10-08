""" Utility programs for plotting.
    V. Ragulin - 10/08/2024
"""
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from tte_optimizer_prop_2drv_qp import State
import config_prop_qp as cfg


def load_pickled_results(file_name: str, data_dir: str | None = None) -> dict:
    """ Load the pickled results from the file
        :param file_name: the name of the file to load
        :param data_dir: the directory where the file is located, if None use the default directory
    """
    if data_dir is None:
        data_dir = os.path.join(current_dir, '..', cfg.SIM_RESULTS_DIR)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'The directory {data_dir} does not exist')

    file_full_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_full_path):
        raise FileNotFoundError(f'The directory {file_full_path} does not exist')

    with open(file_full_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    # A simple test
    FILE_NAME = 'tte_sim_data_l20_r2_o0_01_n20_g0.01_d0.0.pkl'

    data = load_pickled_results(FILE_NAME)
    keys = data.keys()
    print(f"Loaded a dict with the following keys: {keys} ")
