import json
from src.results.summary import best_params
from src.utils import NpEncoder
from src.model.moe import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


try:
    with open('data/results/best_params.json', 'r') as fin:
            exp_info = json.load(fin)
except:
    exp_info = best_params('results')
    with open('data/results/best_params.json', 'w') as fout:
        json.dump(exp_info, fout, indent=3, cls=NpEncoder)
        
