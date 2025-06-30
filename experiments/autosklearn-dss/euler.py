
import os
import sys

sys.path.append("../")

from jobs.jobs import euler
from util import get_dataset_paths


TMP = "tmp/euler_job/"
EULER_SCIRPT = """
#PBS -N p_
#PBS -l select=1:ncpus=2:nodetype=n56:mem=15GB
#PBS -l walltime=24:00:00

module load python/3.9.17
cd /{0}/alcobaca/dynamic_pipeline_search_space
source env/bin/activate
cd source/autosklearn_dss

# the command

"""

NCONFS = "autosklearn"
SEEDS = 1
LOCAL = "lustre"


def run_euler():
    cmd = "python3.9 run.py ../../auto-results/{0}/ {1} {0} {2}\n"
    cmd_paths = get_dataset_paths("../../datasets_test/") 

    os.makedirs(TMP, exist_ok=True)

    cmd_str = ""
    for path in cmd_paths:
        cmd_str += cmd.format(NCONFS, path, SEEDS)

    req_path = TMP + "req.txt"
    job_path = TMP + "job.txt"

    f = open(job_path, "w")
    f.write(cmd_str)
    f.close()

    f = open(req_path, "w")
    f.write(EULER_SCIRPT.format(LOCAL))
    f.close()

    print("Command list:")
    print(cmd_str)
    print()
    print("Requirements list:")
    print(EULER_SCIRPT)
    print()
    euler(command_line=job_path, requirements=req_path, sleep_time=1800,
          job_name="ftm")



if __name__ == "__main__":
    ARGV = sys.argv
    ARGC = len(ARGV)

    if ARGV != 6:
        print("Usage: ")
        print("    python euler_500.py automl seed local")

    NCONFS = str(ARGV[1])
    SEEDS = int(ARGV[2])
    LOCAL = str(ARGV[3])

    run_euler()


