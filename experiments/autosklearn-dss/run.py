import sys
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.append("../")

import json
from sklearn.model_selection import train_test_split
import pandas as pd

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import f1_weighted
from util import read_dataset, create_result_directory
from autosklearn_dss import AutoSklearnClassifierDSS
import shutil


def save_experiment(result_directory, dataset_name, obj):
    df_path = result_directory + dataset_name + "_cv_results.csv"
    df = pd.DataFrame(obj.cv_results_)
    df.to_csv(df_path, index=False)

    df_path = result_directory + dataset_name + "performance_over_time_.csv"
    df = pd.DataFrame(obj.performance_over_time_)
    df.to_csv(df_path, index=False)


def ger_dataset_name(dataset_path):
    return "dataset_" + dataset_path.split("dataset_")[1].split(".pkl")[0]


def ger_directory_name(directory, dataset_name, seed):
    return directory + dataset_name + "/" + str(seed) + "/"


def ger_tmp_folder_name(dataset_name, automl, seed):
    return "tmp-" + dataset_name + "__" + str(automl) + "__" + str(seed) + "/"


def ger_output_folder_name(dataset_name, automl, seed):
    return "output-" + dataset_name + "__" + str(automl) + "__" + str(seed) + "/"


def create_result_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    return


def if_result_directory_exit(directory_name):
    if os.path.exists(directory_name):
        print("Experiment finished")
        exit(0)

    return


def compute_final_performance(automl, X_train, y_train, X_test, y_test, dataset_name, method):
    train = f1_weighted(y_train, automl.predict(X_train))
    val = automl.performance_over_time_["ensemble_optimization_score"].max()
    test = f1_weighted(y_test, automl.predict(X_test))
    return pd.DataFrame({"dataset":dataset_name, "method":method, "train": [train], "val": [val], "test":[test]})


def generate_pipelines(
        dataset_path,
        result_directory,
        time_left_for_this_task=120,
        per_run_time_limit=30,
        memory_limit=10240,
        resampling_strategy="holdout",
        seed=1,
        automl="autosklearn",
):
    dataset_name = ger_dataset_name(dataset_path)
    directory_name = ger_directory_name(result_directory, dataset_name, seed)
    tmp_folder_name = ger_tmp_folder_name(dataset_name, automl, seed)
    output_folder_name =  ger_output_folder_name(dataset_name, automl, seed)

    path_performance = directory_name + dataset_name + "_performance.csv"
    path_csv_results = directory_name + dataset_name + "_cv_results.csv"
    path_performance_over_time = directory_name + dataset_name + "_performance_over_time.csv"
    
    print(directory_name)
    if_result_directory_exit(path_performance_over_time) # pass

    X, y, categorical_indicator, attribute_names = read_dataset(dataset_path)
    y = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=seed)


    estimator = None

    if automl == "autosklearn":
        estimator = AutoSklearnClassifier(
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            initial_configurations_via_metalearning=0,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train)
    
    elif automl == "random_forest":
        estimator = AutoSklearnClassifier(
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            initial_configurations_via_metalearning=0,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            include = {
                'classifier': ["random_forest"],
                'feature_preprocessor': ["no_preprocessing"]
            },
            seed=seed
        )
        estimator.fit(X_train, y_train)
 
    elif automl == "autosklearn_dss_95":
        estimator = AutoSklearnClassifierDSS(
            theta=0.95,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            initial_configurations_via_metalearning=0,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train, categorical_indicator=categorical_indicator, use_cache=dataset_name)
    elif automl == "autosklearn_dss_90":
        estimator = AutoSklearnClassifierDSS(
            theta=0.90,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            initial_configurations_via_metalearning=0,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train, categorical_indicator=categorical_indicator, use_cache=dataset_name)
    elif automl == "autosklearn_dss_90_mtl":
        estimator = AutoSklearnClassifierDSS(
            theta=0.90,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train, categorical_indicator=categorical_indicator, use_cache=dataset_name)
    elif automl == "autosklearn_mtl":
        estimator = AutoSklearnClassifier(
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train)
    elif automl == "autosklearn_dss_95_mtl":
        estimator = AutoSklearnClassifierDSS(
            theta=0.95,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train, categorical_indicator=categorical_indicator, use_cache=dataset_name)
    elif automl == "autosklearn_dss_90_mtl_space":
        estimator = AutoSklearnClassifierDSS(
            theta=0.90,
            time_left_for_this_task=time_left_for_this_task,
            per_run_time_limit=per_run_time_limit,
            memory_limit=memory_limit,
            resampling_strategy=resampling_strategy,
            metric=f1_weighted,
            tmp_folder=tmp_folder_name,
            n_jobs=1,
            delete_tmp_folder_after_terminate=False,
            seed=seed
        )
        estimator.fit(X_train, y_train, categorical_indicator=categorical_indicator, use_cache=dataset_name, only_mtl=True)
        include = estimator.include

        create_result_directory(directory_name)
        path_ = directory_name + dataset_name + "_include.json"
        with open(path_, 'w') as json_file:
            json.dump(include, json_file, indent=4)
    


    df_performance = compute_final_performance(estimator, X_train, y_train, X_test, y_test, dataset_name, automl)
    df_csv_result = pd.DataFrame(estimator.cv_results_)
    df_performance_over_time = pd.DataFrame(estimator.performance_over_time_)

    create_result_directory(directory_name)
    df_performance.to_csv(path_performance, index=False)
    df_csv_result.to_csv(path_csv_results, index=False)
    df_performance_over_time.to_csv(path_performance_over_time, index=False)

    if os.path.exists(tmp_folder_name):
        shutil.rmtree(tmp_folder_name, ignore_errors=True)
        os.rmdir(tmp_folder_name)

# command
# python run.py result_fold seed
if __name__ == "__main__":
    ARGV = sys.argv
    ARGC = len(ARGV)

    if ARGV != 6:
        print("Usage: ")
        print("    python run.py directory dataset_path automl seed")

    directory = ARGV[1]
    dataset_path = ARGV[2]
    automl = str(ARGV[3])
    seed = int(ARGV[4])

    print("directory: ", directory)
    print("dataset_path: ", dataset_path)
    print("automl: ", automl)
    print("seed: ", seed)

    generate_pipelines(
        dataset_path=dataset_path,
        result_directory=directory,
        #time_left_for_this_task=300,
        time_left_for_this_task=3600,  # 1h
        #per_run_time_limit=200,  # 10m
        per_run_time_limit=600,  # 10m
        memory_limit=10240,
        resampling_strategy="holdout",
        seed=seed,
        automl=automl
    )
