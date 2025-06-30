import pickle
import os


def read_dataset(path):
    with open(path, "rb") as f:
        X, y, categorical_indicator, attribute_names = pickle.load(f)
    return X, y, categorical_indicator, attribute_names


def get_dataset_paths(directory):
    dataset_path = []

    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            dataset_path.append(os.path.join(directory, file))
    return dataset_path


def create_result_directory(dataset_path, directory, seed):
    dataset_name = "dataset_" + dataset_path.split("dataset_")[1].split(".pkl")[0]
    directory = directory + "/" + dataset_name + "/" + str(seed)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def save_obj(result_directory, dataset_name, obj_dict):
    for key, obj in obj_dict.items():
        pickle_path = os.path.join(result_directory, f"{dataset_name}_{key}.pkl")

        with open(pickle_path, "wb") as f:
            pickle.dump(obj, f)
