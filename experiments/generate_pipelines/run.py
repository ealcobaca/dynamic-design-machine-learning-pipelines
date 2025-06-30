import sys

from generate_pipelines import generate_pipelines, metric_list

import os

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


# command
# python run.py result_fold seed
if __name__ == "__main__":
    ARGV = sys.argv
    ARGC = len(ARGV)

    if ARGV != 6:
        print("Usage: ")
        print("    python run.py directory dataset_path number_iterations seed")

    directory = ARGV[1]
    dataset_path = ARGV[2]
    confs = int(ARGV[3])
    seed = int(ARGV[4])

    print("directory: ", directory)
    print("dataset_path: ", dataset_path)
    print("nconf: ", confs)
    print("seed: ", seed)

    generate_pipelines(
        dataset_path=dataset_path,
        result_directory=directory,
        time_left_for_this_task=86400,  # 12h
        per_run_time_limit=600,  # 10m
        memory_limit=10240,
        resampling_strategy="holdout",
        seed=seed,
        number_of_configs=confs
    )
