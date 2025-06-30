import sys

from get_dataset_from_openml import download_datasets, get_id_list_from_tasks


if __name__ == "__main__":

    download_datasets(
        directory="../../datasets/",
        id_list=get_id_list_from_tasks("../../datasets/datasets_meta_data.csv")
    )


