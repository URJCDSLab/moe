from sklearnex import patch_sklearn
patch_sklearn()

import json
import pandas as pd
from src.results.summary import best_params
from src.utils import NpEncoder, Timer
from src.model.moe import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier


try:
    with open('data/results/best_params.json', 'r') as fin:
            params_info = json.load(fin)
except:
    params_info = best_params('results')
    with open('data/results/best_params.json', 'w') as fout:
        json.dump(params_info, fout, indent=3, cls=NpEncoder)


models = {'MoeSvm':{'model': MOE,
                    'kwargs':{'method':SVC, 'params':{'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}},
                    },
          'MoeKnn':{'model': MOE,
                    'kwargs':{'method':KNeighborsClassifier, 'params':{'n_neighbors' : list(range(1, 13, 2))}},
                    },
          'MoeDt':{'model': MOE,
                    'kwargs':{'method':DecisionTreeClassifier, 'params':{'criterion' : {"gini", "entropy"}, 'max_depth' : list(range(1, 10, 1))}},
                    },
          'RandomForest':{'model': RandomForestClassifier},
          'SVM':{'model': SVC},
          'KNN':{'model': KNeighborsClassifier},
          'XGBoost':{'model': XGBClassifier},
          'GradientBoosting':{'model': GradientBoostingClassifier},
          'ExtraTrees':{'model':ExtraTreesClassifier}}


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
     
    data = pd.read_parquet(f'data/prep_real_data/{experiment}.parquet')


    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values
    
    exp_info = {}
    
    for model in models.keys():
        print('Training... ', model)
        try:
            clf =  models[model]['model'](**params_info[experiment][model]['params'], **models[model]['kwargs'])
        except:
            clf =  models[model]['model'](**params_info[experiment][model]['params'])
        
        with Timer(model) as timer:
            clf.fit(X, y)
        
        seconds = timer.interval
        
        exp_info[model] = seconds
        
    print(exp_info)
    
    with open(f'data/results/times/{experiment}.json', 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)
    
