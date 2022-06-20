from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import pickle

from src.model.moe import *
from src.utils import *

import warnings
warnings.filterwarnings('ignore')


for experiment in [
'appendicitis',
'australian',
'backache',
'banknote',
'breastcancer',
'bupa',
'cleve',
'colon-cancer',
'diabetes',
'flare',
'fourclass',
'german_numer',
'haberman',
'heart',
'ilpd',
'ionosphere',
'kr_vs_kp',
'liver-disorders',
'lupus',
'mammographic',
'mushroom',
'r2',
'svmguide1',
'svmguide3',
'transfusion'
 ]:


    print(f'Experiment: {experiment}\n')

    results_folder = f'results/{experiment}'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/prep_real_data/{experiment}.parquet')


    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values


    # SVM hyperparameters gridsearch

    grid_params = {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf'], 'random_state': [1234]}

    svm = gridsearch(SVC, grid_params, scaled_mcc, cv=10, n_jobs=-1)

    print('SVM: \n')

    svm.fit(X, y)

    print(f'params: {svm.best_params_}, score: {svm.best_score_}\n')

    with open(f'results/{experiment}/SVM.p', 'wb') as fout:
        pickle.dump(svm, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # KNN hyperparameters gridsearch

    grid_params = {'n_neighbors' : list(range(1, 13, 2)), 'n_jobs': [-1]}

    knn = gridsearch(KNeighborsClassifier, grid_params, scaled_mcc, cv=10, n_jobs=-1)

    print('KNN: \n')

    knn.fit(X, y)   

    print(f'params: {knn.best_params_}, score: {knn.best_score_}\n')

    with open(f'results/{experiment}/KNN.p', 'wb') as fout:
        pickle.dump(knn, fout, protocol=pickle.HIGHEST_PROTOCOL)


    # MOE-kNN hyperparameters gridsearch
    grid_params = {
        'wrab': [True, False],
        'lam': [1, 3, 5],
        'prop_sample': [0.10, 0.30, 0.50],
        'n_learners': [10, 20, 30],
        'random_state': [1234]
    }

    kwargs = {'method':KNeighborsClassifier, 'params':{'n_neighbors' : list(range(1, 13, 2))}}

    moe_knn = MOE

    moe_knn_grid = gridsearch(moe_knn, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

    print('MOE kNN: \n')

    moe_knn_grid.fit(X, y)

    print(f'params: {moe_knn_grid.best_params_}, score: {moe_knn_grid.best_score_}\n')

    with open(f'results/{experiment}/MoeKnn.p', 'wb') as fout:
        pickle.dump(moe_knn_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)
        


    # MOE-DT hyperparameters gridsearch
    grid_params = {
        'wrab': [True, False],
        'lam': [1, 3, 5],
        'prop_sample': [0.10, 0.30, 0.50],
        'n_learners': [10, 20, 30],
        'random_state': [1234]
    }

    kwargs = {'method':DecisionTreeClassifier, 'params':{'criterion' : {"gini", "entropy"}, 'max_depth' : list(range(1, 10, 1))}}

    moe_dt = MOE

    moe_dt_grid = gridsearch(moe_dt, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

    print('MOE DT: \n')

    moe_dt_grid.fit(X, y)

    print(f'params: {moe_dt_grid.best_params_}, score: {moe_dt_grid.best_score_}\n')

    with open(f'results/{experiment}/MoeDt.p', 'wb') as fout:
        pickle.dump(moe_dt_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # MOE-SVM hyperparameters gridsearch
    grid_params = {
        'wrab': [True, False],
        'lam': [1, 3, 5],
        'prop_sample': [0.10, 0.30, 0.50],
        'n_learners': [10, 20, 30],
        'random_state': [1234]
    }

    kwargs = {'method':SVC, 'params':{'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}}

    moe_svm = MOE

    moe_svm_grid = gridsearch(moe_svm, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

    print('MOE SVM: \n')

    moe_svm_grid.fit(X, y)

    print(f'params: {moe_svm_grid.best_params_}, score: {moe_svm_grid.best_score_}\n')

    with open(f'results/{experiment}/MoeSvm.p', 'wb') as fout:
        pickle.dump(moe_svm_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)
        
    # XGBoost hyperparameters gridsearch
    grid_params = {'max_depth': [3, 5, 7], 'eta': [0.1, 0.2,
                                    0.3], 'n_estimators': [100, 300, 500],
                'eval_metric': ['logloss'], 'n_jobs': [-1], 'random_state': [1234]}

    xgb = gridsearch(XGBClassifier, grid_params,
                    scaled_mcc, cv=10, n_jobs=-1)

    print('XGBoost: \n')

    xgb.fit(X, y)

    print(f'params: {xgb.best_params_}, score: {xgb.best_score_}\n')

    with open(f'results/{experiment}/XGBoost.p', 'wb') as fout:
        pickle.dump(xgb, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # Extra Trees hyperparameters gridsearch
    grid_params = {'max_features': [None, 'sqrt',
                                    'log2'], 'n_estimators': [100, 300, 500], 'n_jobs': [-1], 'random_state': [1234]}

    et = gridsearch(ExtraTreesClassifier, grid_params,
                    scaled_mcc, cv=10, n_jobs=-1)

    print('Extra Trees: \n')

    et.fit(X, y)

    print(f'params: {et.best_params_}, score: {et.best_score_}\n')

    with open(f'results/{experiment}/ExtraTrees.p', 'wb') as fout:
        pickle.dump(et, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # Gradient Boosting hyperparameters gridsearch
    grid_params = {'max_features': [None, 'sqrt',
                                    'log2'], 'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.2,
                                    0.3], 'n_estimators': [100, 300, 500], 'random_state': [1234]}

    gb = gridsearch(GradientBoostingClassifier, grid_params,
                    scaled_mcc, cv=10, n_jobs=-1)

    print('Gradient Boosting: \n')

    gb.fit(X, y)

    print(f'params: {gb.best_params_}, score: {gb.best_score_}\n')

    with open(f'results/{experiment}/GradientBoosting.p', 'wb') as fout:
        pickle.dump(gb, fout, protocol=pickle.HIGHEST_PROTOCOL)

    # Random Forest hyperparameters gridsearch
    grid_params = {'max_features': [None, 'sqrt',
                                    'log2'], 'n_estimators': [100, 300, 500], 'random_state': [1234]}

    rf = gridsearch(RandomForestClassifier, grid_params,
                    scaled_mcc, cv=10, n_jobs=-1)

    print('Random Forest: \n')

    rf.fit(X, y)

    print(f'params: {rf.best_params_}, score: {rf.best_score_}\n')

    with open(f'results/{experiment}/RandomForest.p', 'wb') as fout:
        pickle.dump(rf, fout, protocol=pickle.HIGHEST_PROTOCOL)
    