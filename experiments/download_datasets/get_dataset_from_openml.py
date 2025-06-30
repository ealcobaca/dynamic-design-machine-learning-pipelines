import pickle
import os

import pandas as pd

import openml


def make_dataset_description(directory, id_list):
    df_dataset_description = openml.datasets.list_datasets(id_list, output_format="dataframe")
    df_dataset_description.to_csv(f"{directory}dataset_description.csv", index=False)

def pickleing_datasets(directory, id_list, verbose=True):
    id_list = [str(i) for i in id_list]
    
    for i in id_list:
        if verbose:
            print(f"Searching dataset... {i}")
            
        dataset = openml.datasets.get_dataset(i)
        data = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        

        dataset_name = f"{directory}dataset_{i}.pkl"
        if verbose:
            print(f"Saving... {dataset_name}")
            
        with open(dataset_name, 'wb') as f:
            pickle.dump(data, f)

def download_datasets(directory, id_list):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    make_dataset_description(directory, id_list)
    pickleing_datasets(directory, id_list)

def read_dataset(path):
    with open(path, 'rb') as f:
        X, y, categorical_indicator, attribute_names = pickle.load(f)
    return X, y, categorical_indicator, attribute_names
        

def get_id_list_from_tasks(path):
    df_sets = pd.read_csv(path)
    id_list = df_sets["OpenML ID"].astype(int).tolist()

    return id_list
