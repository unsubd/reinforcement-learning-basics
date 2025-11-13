import numpy as np


def export_numpy_array(q_table, dest_file_path):
    if not dest_file_path.endswith(('.npy', '.npz')):
        dest_file_path += '.npy'

    np.save(dest_file_path, q_table)
    print("Q-Table saved to", dest_file_path)


def load_numpy_array(numpy_dump_file):
    return np.load(numpy_dump_file)
