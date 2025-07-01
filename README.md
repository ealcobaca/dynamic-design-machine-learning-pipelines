# Dynamic Design of Machine Learning Pipelines via Metalearning

Paper Title: Dynamic Design of Machine Learning Pipelines via Metalearning

Abstract: Automated machine learning (AutoML) has democratized the design of machine learning based systems, by automating model selection, hyperparameter tuning and feature engineering.
However, the high computational cost associated with traditional search and optimization strategies, such as Random Search, Particle Swarm Optimization and Bayesian Optimization, remains a significant challenge.
Moreover, AutoML systems typically explore a large search space, which can lead to overfitting.
This paper introduces a metalearning method for dynamically designing search spaces for AutoML system.
The proposed method uses historical metaknowledge to select promising regions of the search space, accelerating the optimization process.
According to experiments conducted for this study, the proposed method can reduce runtime by 89\% in Random Search and search space by (1.8/13 preprocessor and 4.3/16 classifier), without compromising significant predictive performance.
Moreover, the proposed method showed competitive performance when adapted to Auto-Sklearn, reducing its search space.
Furthermore, this study encompasses insights into meta-feature selection, meta-model explainability, and the trade-offs inherent in search space reduction strategies.


## Repository Organization

This repository is organized as follow:

```markdown
- analysis/
  - meta-level-analysis/
  - autosklearn-dss/
  - dataset-analysis/
  - dynamic-seach space/
  - meta-feature-analysis/
  - meta-model-design/
  - README.md
  - requirements.txt
- experiments/
  -  generate_pipelines/
  -  download_datasets/
  -  autosklearn-dss/
  -  requirements.txt
  -  README.md
- datasets/
  - datasets_meta_data.csv
- paper/
```

## Run

* For reproduce experiment you can use Makefile for run/test:
  - clean : clean all log data
  - create-env-experiment : create env to run RS experiments
  - run-example : run RS experiment example
  - run-experiment : run RS experiment
  - create-env-experiment-dss : create a env to run Auto-Sklearn with dynamic search space experiment
  - run-example-dss : run Auto-Sklearn with dynamic search space experiment example
  - run-experiment-dss : run Auto-Sklearn with dynamic search space experiment

* For the analysis part, you should:
    - Download the results scripts from this [link]().
    - Create a Python3.10 env with `requiriments_colab.txt`
    - Run the notebooks
