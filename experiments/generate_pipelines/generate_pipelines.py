import os
import sys

sys.path.append("../")

from sklearn.model_selection import train_test_split
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import accuracy, balanced_accuracy, f1_macro, f1_weighted, precision_macro, precision_weighted, \
    recall_macro, recall_weighted
from sklearn.model_selection import StratifiedKFold

from util import read_dataset, create_result_directory

import hashlib
import json


def dict_hash(dictionary):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def metric_list():
    metrics = [
        accuracy,
        balanced_accuracy,
        f1_macro,
        f1_weighted,
        precision_macro,
        precision_weighted,
        recall_macro,
        recall_weighted
    ]
    return metrics


def apply_metrics(y_true, y_pred, tag):
    return {f.name + "_" + tag: f(y_true, y_pred) for f in metric_list()}


def update_dicts(dict_list):
    if len(dict_list) == 0:
        return {}

    d = dict_list.pop()
    result = update_dicts(dict_list)
    result.update(d)
    return result


def save_experiment(result_directory, dataset_name, obj):
    df_path = result_directory + "/" + dataset_name + "_cv_results_.csv"
    df = pd.DataFrame(obj.cv_results_)
    df.to_csv(df_path, index=False)

    df_path = result_directory + "/" + dataset_name + "performance_over_time_.csv"
    df = pd.DataFrame(obj.performance_over_time_)
    df.to_csv(df_path, index=False)


def ger_dataset_name(dataset_path):
    return "dataset_" + dataset_path.split("dataset_")[1].split(".pkl")[0]


def ger_directory_name(directory, dataset_name, seed):
    return directory + "/" + dataset_name + "/" + str(seed)


def ger_tmp_fold_name(dataset_name, time_left_for_this_task, seed):
    return "auto-sklearn-" + dataset_name + "_" + str(time_left_for_this_task) + "_" + str(seed)


def create_result_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    return


def if_result_directory_exit(directory_name):
    if os.path.exists(directory_name):
        print("Experiment finished")
        exit(0)

    return


def generate_pipelines(
        dataset_path,
        result_directory,
        time_left_for_this_task=120,
        per_run_time_limit=30,
        memory_limit=10240,
        resampling_strategy="holdout",
        seed=1,
        number_of_configs=2,
        n_splits=10
):
    dataset_name = ger_dataset_name(dataset_path)
    directory_name = ger_directory_name(result_directory, dataset_name, seed)
    tmp_folder_name = ger_tmp_fold_name(dataset_name, number_of_configs, seed)

    #     print(directory_name)
    #     if_result_directory_exit(directory_name) # pass

    X, y, categorical_indicator, attribute_names = read_dataset(dataset_path)
    y = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=seed)

    estimator = AutoSklearnClassifier(
        time_left_for_this_task=time_left_for_this_task,
        per_run_time_limit=per_run_time_limit,
        memory_limit=memory_limit,
        initial_configurations_via_metalearning=0,
        resampling_strategy=resampling_strategy,
        scoring_functions=metric_list(),
        tmp_folder=tmp_folder_name,
        delete_tmp_folder_after_terminate=False,
        seed=seed,
        ensemble_class=None
    )

    # generate configs
    cs = estimator.get_configuration_space(X_train, y_train, dataset_name=dataset_name)
    cs.seed(seed)
    configs = cs.sample_configuration(number_of_configs)

    # result dir
    create_result_directory(directory_name)

    for config, config_id in zip(configs, range(len(configs))):

        # open file to save the results
        df_path = directory_name + "/" + dataset_name + "_cv_results_iter.csv"
        result_df = pd.read_csv(df_path) if os.path.exists(df_path) else None

        # if exist a related config id
        config_hash = dict_hash(config.get_dictionary())
        if isinstance(result_df, pd.DataFrame):
            if config_hash in result_df["config_hash"].unique():
                continue

        try:
            config.is_valid_configuration()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            i = 0
            for train_index, test_index in skf.split(X_train, y_train):
                i += 1
                X_train_i, y_train_i = X_train.iloc[train_index, :], y_train.iloc[train_index]
                X_test_i, y_test_i = X_train.iloc[test_index, :], y_train.iloc[test_index]

                pipeline, run_info, run_value = estimator.fit_pipeline(
                    X=X_train_i,
                    y=y_train_i,
                    dataset_name=dataset_name,
                    config=config,
                    X_test=X_test_i,
                    y_test=y_test_i,
                )

                result_dict = {
                    "seed_i": seed,
                    "config_id": config_id,
                    "fold": i,
                    "config_hash": config_hash
                }
                result_dict.update({
                    "duration": run_value.time,
                    'start_time': run_value.starttime,
                    'end_time': run_value.endtime,
                    'status': str(run_value.status)
                })
                result_dict.update({
                    "seed": run_info.seed,
                    "budget": run_info.budget,
                })
                result_dict.update(run_info.config.get_dictionary())

                if pipeline != None:
                    Xs = [X_train_i, X_test_i, X_test]
                    ys = [y_train_i, y_test_i, y_test]
                    tags = ["train", "val", "test"]
                    yps = [pipeline.predict(Xi) for Xi in Xs]
                    perfs = [apply_metrics(yt, yp, t) for yt, yp, t in zip(ys, yps, tags)]
                    result_dict.update(update_dicts(perfs))

                result_frame = pd.Series(result_dict).to_frame().T.reset_index()

                result_df = pd.concat([result_df, result_frame]) if isinstance(result_df,
                                                                               pd.DataFrame) else result_frame

            if (config_id % 10) == 0:
                result_df.to_csv(df_path, index=False)

        except Exception as e:
            print(e)
            continue

        result_df.to_csv(df_path, index=False)

    import shutil
    shutil.rmtree(tmp_folder_name, ignore_errors=True)
