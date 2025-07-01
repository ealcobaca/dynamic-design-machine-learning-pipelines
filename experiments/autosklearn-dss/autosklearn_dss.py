import os.path
import numpy as np
import pickle
import pandas as pd
from pymfe.mfe import MFE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from autosklearn.metrics import f1_weighted

import sklearn.datasets
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from autosklearn.classification import AutoSklearnClassifier


METATARGET = 'f1_weighted_test'

MTF = [
 'nodes_per_inst',
 'joint_ent',
 'nr_cat',
 'class_ent',
 'attr_ent',
 'nr_num',
 'w_lambda',
 'var_importance',
 'attr_to_inst',
 'nr_attr',
 'freq_class'
]

SUMM = ["mean", "sd"]

METAFEATURES = ['nodes_per_inst', 'p_mean_perf', 'p_max_perf', 'p_median_perf', 'p_min_perf',
 'joint_ent.mean', 'p_std_perf', 'nr_cat', 'class_ent', 'attr_ent.mean',
 'pr_std_perf', 'preprocessor-num', 'classifier-num', 'pr_median_perf',
 'pr_mean_perf', 'cl_median_perf', 'cl_std_perf', 'pr_min_perf', 'cl_mean_perf',
 'nr_num', 'w_lambda', 'var_importance.sd', 'attr_to_inst', 'nr_attr',
 'freq_class.sd']

PIP_METAFEATURES = [
 'cl_min_perf',
 'cl_max_perf',
 'cl_median_perf',
 'cl_mean_perf',
 'cl_std_perf',
 'pr_min_perf',
 'pr_max_perf',
 'pr_median_perf',
 'pr_mean_perf',
 'pr_std_perf',
 'p_min_perf',
 'p_max_perf',
 'p_median_perf',
 'p_mean_perf',
 'p_std_perf',
 'classifier-num',
 'preprocessor-num'
]

SEED = 10


class AutoSklearnClassifierDSS(AutoSklearnClassifier):
    def __init__(self, theta=0.95, **kwargs):

        self.theta=theta
        super().__init__(**kwargs)


    def imput_missing_values(cls, X):
        fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype.name == 'category' else X[c].mean() for c in X], index=X.columns)
        return X.fillna(fill)


    def preprocessing_data(cls, X, y, categorical_indicator):
        cat_cols = "auto"
        if np.any(categorical_indicator):
            cat_cols = np.arange(0, X.shape[1])[categorical_indicator]

        X = cls.imput_missing_values(X)

        return X.to_numpy(), y.to_numpy(), cat_cols

    
    def compute_meta_features_cached(cls, dataset):
        D_TEST_META = pd.read_csv("d_test_meta.csv")
        query = f"dataset == '{dataset}'"
        mtf_ = D_TEST_META.query(query)
        return mtf_

    def fit(self, X, y, X_test=None, y_test=None, feat_type=None, dataset_name=None, categorical_indicator=None, use_cache=None, only_mtl=False):
        ## meta-feature extraction
        
        if use_cache == None:
            categorical_indicator = [] if categorical_indicator == None else categorical_indicator
            X_, y_, cat_cols = self.preprocessing_data(X, y, categorical_indicator)
            meta_dataset = self.compute_meta_dataset(X_, y_, cat_cols)
        else:
            meta_dataset = self.compute_meta_features_cached(use_cache)

        ## meta-model prediction
        meta_model = self.get_meta_model()
        predictions = meta_model.predict(meta_dataset[METAFEATURES])
        meta_dataset = meta_dataset.assign(prediction=predictions)

        qt = meta_dataset["prediction"].quantile([self.theta]).iloc[0]
        mask = meta_dataset["prediction"] > qt
        meta_dataset = meta_dataset[mask]

        preprocessors = meta_dataset["preprocessor"].unique().tolist()
        classifiers = meta_dataset["classifier"].unique().tolist()

        self.include = {
                    'classifier': classifiers,
                    'feature_preprocessor': preprocessors
                }
        
        print(self.include)

        if only_mtl:
            return

        return super().fit(X, y, X_test, y_test, feat_type, dataset_name)

    def compute_meta_dataset(self, X_train, y_train, cat_cols):
        minn = -1e10
        maxx = 1e10

        meta_features = self.get_standard_meta_features(X_train, y_train, cat_cols)
        pip_meta_features = self.get_pipeline_performance_meta_features()
        meta_dataset = pip_meta_features.join(meta_features, how="cross")\
            .replace([np.inf], maxx)\
            .replace([-np.inf], minn)\
            .fillna(0.0)
        return meta_dataset


    def get_standard_meta_features(self, X_train, y_train, cat_cols):
        mfe = MFE(features=MTF, summary=SUMM)
        meta_features = mfe.fit(X_train, y_train).extract(cat_cols=cat_cols, suppress_warnings=True)
        meta_features = pd.DataFrame({k:[v] for k, v in zip(*meta_features)})
        return meta_features


    def get_pipeline_performance_meta_features(cls):
        filename = "pipeline_perforamnce_meta_features.csv"
        return pd.read_csv(filename)


    def get_meta_model(cls):
      model_path="meta-model.plk"
      meta_dataset_path="D_train_meta.csv"
      meta_dataset_path_test = "d_test_meta.csv"

      if  os.path.isfile(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
      else:
          df = pd.read_csv(meta_dataset_path)
          df_test = pd.read_csv(meta_dataset_path_test)
          X_train = df[METAFEATURES]
          y_train = df[METATARGET]
          X_test = df_test[METAFEATURES]
          y_test = df_test[METATARGET]
            
            
          model = Pipeline([('RF', TransformedTargetRegressor(regressor=RandomForestRegressor(
              n_estimators=1000, 
              random_state=SEED
          ), func=np.log1p, inverse_func=np.expm1))])
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
        
          print("R2: ", r2_score(y_test, y_pred))
          print("RMSE: ",mean_squared_error(y_test, y_pred)**0.5)
          print("MSE: ", mean_squared_error(y_test, y_pred))
          print("MAE: ", mean_absolute_error(y_test, y_pred))
          with open(model_path, 'wb') as file:
            pickle.dump(model, file)

      return model

                                                                                             
