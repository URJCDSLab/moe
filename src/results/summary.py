import os
import pandas as pd
import pickle
import itertools


model_names = ['MoeSvm.p', 'MoeKnn.p', 'MoeDt.p','RandomForest.p', 'SVM.p', 'KNN.p', 'XGBoost.p', 'GradientBoosting.p', 'ExtraTrees.p']
methods = ['MOE SVM', 'MOE KNN', 'MOE DT','Random Forest', 'SVM', 'KNN', 'XGBoost', 'Gradient Boosting', 'Extra Trees']

def name_exps(results_path):
    exp_data = os.listdir(results_path)
    exp_data = [i for i in exp_data if i != '.gitkeep']
    return sorted(exp_data)
    

def df_description(results_path='../results', df_path='../data/prep_real_data'):
    exp_data = name_exps(results_path)
    dfs = pd.DataFrame(columns=['instances', 'n_features', 'class_prop'], index=exp_data, data = [])

    for exp in exp_data:
        try:
            X = pd.read_parquet(f'{df_path}/{exp}.parquet')
            dfs.loc[exp, 'instances'] = X.shape[0]
            dfs.loc[exp, 'n_features'] = X.shape[1]
            dfs.loc[exp, 'class_prop'] = round(min(X['y'].value_counts()/X.shape[0]), 3)
        except :
            pass
    dfs.sort_index(inplace=True)
    return dfs

def mean_df(results_path='../results', model_names=model_names, methods=methods):
    
    exp_data = name_exps(results_path)
    df_mean = pd.DataFrame(index = exp_data, columns = methods)
    
    for exp in exp_data:
        scores = []
        for model in model_names:
            try:
                with open('/'.join(['/'.join([results_path, exp]), model]), 'rb') as fin:
                        scores.append(round(pickle.load(fin).best_score_[0], 3))
            except:
                scores.append(None)
        df_mean.loc[exp, :] = scores
    return df_mean

def sd_df(results_path='../results', model_names=model_names, methods=methods):
    
    exp_data = name_exps(results_path)
    df_sd = pd.DataFrame(index = exp_data, columns = methods)
    
    for exp in exp_data:
        scores = []
        for model in model_names:
            try:
                with open('/'.join(['/'.join([results_path, exp]), model]), 'rb') as fin:
                        scores.append(round(pickle.load(fin).best_score_[1], 3))
            except:
                scores.append(None)
        df_sd.loc[exp, :] = scores
    return df_sd

def mean_sd_df(results_path='../results', model_names=model_names, methods=methods):
    
    exp_data = name_exps(results_path)
    df_results = pd.DataFrame(index = exp_data, columns = methods)
    
    for exp in exp_data:
        scores = []
        for model in model_names:
            try:
                with open('/'.join(['/'.join([results_path, exp]), model]), 'rb') as fin:
                        scores.append(' ('.join([str(round(elem, 3))  for elem in pickle.load(fin).best_score_] ) + ')')
            except:
                scores.append(None)
        df_results.loc[exp, :] = scores
    return df_results

def rank_df(df_mean, method='MOE SVM', methods=methods):
    
    rank_index = [i for i in methods if i != method]
    wtl_df = pd.DataFrame(index = rank_index, 
                               columns = ['Wins', 'Ties', 'Losses'])
    for i in rank_index:
        wtl_df.loc[i, 'Wins'] = sum(df_mean[method] > df_mean[i])
        wtl_df.loc[i, 'Losses'] = sum(df_mean[method] < df_mean[i])
        wtl_df.loc[i, 'Ties'] = len(df_mean) - wtl_df.loc[i, 'Wins'] - wtl_df.loc[i, 'Losses']
    wtl_plot = wtl_df.copy()
    wtl_plot['Ties'] = wtl_df['Wins'] + wtl_df['Ties']
    wtl_plot['Losses'] = wtl_plot['Ties'] + wtl_df['Losses']
    
    return wtl_df, wtl_plot

def rank_matrix(df_mean, proportion=False, methods=methods):
    col_comb = list(itertools.combinations(df_mean.columns.values, 2))
    win_loss_df = pd.DataFrame(index = methods, 
                               columns = methods)
    if proportion==True:
        prop=df_mean.shape[0]
        tot = 1
        for i, j in col_comb:
            win = round(sum(df_mean[i] > df_mean[j])/prop, 2)
            loss = round(sum(df_mean[i] < df_mean[j])/prop, 2)
            tie = round(tot- win - loss, 2)
            win_loss_df.loc[i, j] = '-'.join([str(win), str(tie), str(loss)])
    else:
        tot = len(df_mean)
        for i, j in col_comb:
            win = sum(df_mean[i] > df_mean[j])
            loss = sum(df_mean[i] < df_mean[j])
            tie = tot- win - loss
            win_loss_df.loc[i, j] = '-'.join([str(win), str(tie), str(loss)])
    return win_loss_df

def params(results_path='../results', method='MoeSvm.p'):
    exp_data = name_exps(results_path)
    
    params_df = pd.DataFrame(index = exp_data, columns=['WRAB', 'lambda', 'N. Learners', 'P. Sample'])
    
    for exp in exp_data:
        with open('/'.join(['/'.join([results_path, exp]), method]), 'rb') as fin:
            params = pickle.load(fin).best_params_
            params_df.loc[exp, 'WRAB'] = params['wrab']
            params_df.loc[exp, 'lambda'] = params['lam']
            params_df.loc[exp, 'N. Learners'] = params['n_learners']
            params_df.loc[exp, 'P. Sample'] = params['prop_sample']
    params_df['lambda'] = params_df['lambda'].astype(int)
    params_df['N. Learners'] = params_df['N. Learners'].astype(int)
    params_df['P. Sample'] = params_df['P. Sample'].astype(float)
    return params_df

def hyperparameters_summary(results_path='results'):
    params_df_svm = params(results_path=results_path, method='MoeSvm.p')
    params_df_dt = params(results_path=results_path, method='MoeDt.p')
    params_df_knn = params(results_path=results_path, method='MoeKnn.p')
    
    moe_svm_params = pd.concat([params_df_svm.groupby(['WRAB']).size(), params_df_svm.groupby(['lambda']).size(), params_df_svm.groupby(['N. Learners']).size(), params_df_svm.groupby(['P. Sample']).size()])
    moe_dt_params = pd.concat([params_df_dt.groupby(['WRAB']).size(), params_df_dt.groupby(['lambda']).size(), params_df_dt.groupby(['N. Learners']).size(), params_df_dt.groupby(['P. Sample']).size()])
    moe_knn_params = pd.concat([params_df_knn.groupby(['WRAB']).size(), params_df_knn.groupby(['lambda']).size(), params_df_knn.groupby(['N. Learners']).size(), params_df_knn.groupby(['P. Sample']).size()])
    
    index = pd.MultiIndex.from_tuples([
            ('Wrab', 'False'),
            ('Wrab', 'True'),
            ('lambda', '1'),
            ('lambda', '3'),
            ('lambda', '5'),
            ('N. Learners', '10'),
            ('N. Learners', '20'),
            ('N. Learners', '30'),
            ('P. Sample', '0.10'),
            ('P. Sample', '0.30'),
            ('P. Sample', '0.50')],  names=['type', 'value'])
    
    hyperparameters_summary = pd.DataFrame(data=[moe_svm_params.values, moe_dt_params.values, moe_knn_params.values], columns =  index, index = ['MOE SVM', 'MOE DT', 'MOE KNN']).transpose()
    return hyperparameters_summary


def best_params(results_path='../results'):
    exps = name_exps(results_path)
    best_params = {}
    for exp in exps:
        best_params[exp] = {}
        for model in model_names:
            best_params[exp][model[:-2]] = {}
            with open('/'.join(['/'.join([results_path, exp]), model]), 'rb') as fin:
                best_params[exp][model[:-2]]['params'] = pickle.load(fin).best_params_
    return best_params