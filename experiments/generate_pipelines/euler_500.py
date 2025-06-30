
import os
import sys

sys.path.append("../")

from jobs.jobs import euler
from util import get_dataset_paths


TMP = "tmp/euler_job/"
EULER_SCIRPT = """
#PBS -q alcobaca
#PBS -N p_
#PBS -l select=1:ncpus=2:nodetype=n40:mem=10GB
#PBS -l walltime=24:00:00

module load python/3.10.1-2 
cd /{0}/alcobaca/dynamic_pipeline_search_space
source env3.10/bin/activate
cd source/generate_pipelines 

# the command

"""

NCONFS = "1"
SEEDS = 1
LOCAL = "lustre"


def run_euler():
    cmd = "python3.10 run.py ../../results/pipeline_generation/{0}iter {1} {0} {2}\n"
    cmd_paths = get_dataset_paths("../../datasets/") 

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
        print("    python euler_500.py number_iterations seed local")

    NCONFS = str(ARGV[1])
    SEEDS = int(ARGV[2])
    LOCAL = ARGV[3]

    run_euler()


