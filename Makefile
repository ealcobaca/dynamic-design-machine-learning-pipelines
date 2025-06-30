clean:
	@rm -rf experiments/generate_pipelines/auto-sklearn-dataset_*
	@rm -rf experiments/generate_pipelines/p_.*
	@rm -rf experiments/generate_pipelines/tmp


sync-server:
	rsync -azhP $(LOCAL)$(FOLDER)* $(SERVER)$(FOLDER)

sync-local:
	rsync -azhP $(SERVER)$(FOLDER)* $(LOCAL)$(FOLDER)

monitor:
	clear;	while :;  do totaljob; cotas; sleep 30; clear; done

load:
	module load python/3.10.1-2 

cs:
	cotas

tj:
	totaljob

cn:
	clusternodes


create-env-experiment:
	rm -rf env3.10/
	python3.10 -m venv env3.10/
	(\
		source env3.10/bin/activate; \
		pip3 install numpy==1.26.4 pandas==1.5.3 openml==0.15.1 scipy==1.10.1 auto-sklearn==0.15.0; \
	)

download-datasets:
	( \
		source env3.10/bin/activate; \
		cd experiments/download_datasets/ ;\
		python3.10 run.py; \
	)

run-example:
	( \
		source env3.10/bin/activate; \
		cd experiments/generate_pipelines/; \
		export OPENBLAS_NUM_THREADS=1; \
		rm -rf auto-sklearn-dataset_11_2_42/; \
		python3.10 run.py ../../results/pipeline_generation/2iter ../../datasets/dataset_11.pkl 2 42 \
	)

run-experiment-test: clean
	( \
		source env3.10/bin/activate; \
		cd experiments/generate_pipelines/; \
		export OPENBLAS_NUM_THREADS=1; \
		python3.10 euler_500.py 2 42 home \
	)

run-experiment:
	( \
		source env3.10/bin/activate; \
		cd experiments/generate_pipelines2/; \
		export OPENBLAS_NUM_THREADS=1; \
		python3.10 euler_500.py 500 $(seed) home \
	)


create-env-experiment-dss:
	rm -rf env/
	python3.10 -m venv env/
	(\
		source env/bin/activate; \
		pip3 install numpy==1.24.4 pandas==1.5.3 openml==0.15.1 scipy==1.8.0 pymfe==0.4.2 auto-sklearn==0.15.0; \
	)

run-experiment-dss-test:
	( \
		source env/bin/activate; \
		cd experiments/autosklearn_dss/ ; \
		export OPENBLAS_NUM_THREADS=1; \
		python3 run.py ../../auto-results/$(automl)/ ../../datasets_test/dataset_11.pkl $(automl) $(seed)  \
	)
	
run-experiment-dss:
	( \
		source env/bin/activate; \
		cd experiments/autosklearn_dss/ ; \
		export OPENBLAS_NUM_THREADS=1; \
		python3 euler.py autosklearn_dss_90_mtl_space $(seed) home; \
		python3 euler.py autosklearn_dss_95_mtl $(seed) home; \
		python3 euler.py autosklearn_mtl $(seed) home; \
		python3 euler.py random_forest $(seed) home; \
		python3 euler.py autosklearn_dss_90_mtl $(seed) home; \
		python3 euler.py autosklearn $(seed) home; \
		python3 euler.py autosklearn_dss_95 $(seed) home; \
		python3 euler.py autosklearn_dss_90 $(seed) home; \
	)

run-example-dss:
	( \
		source env/bin/activate; \
		cd experiments/autosklearn-dss/ ; \
		export OPENBLAS_NUM_THREADS=1; \
		python3.10 run.py ../../auto-results/ ../../datasets/dataset_11.pkl autosklearn_dss_90_mtl 42 \
	)

