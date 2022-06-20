from src.results.summary import *

dfs = df_description(results_path='results', df_path='data/prep_real_data')
dfs.to_parquet('data/results/dfs_description.parquet')

dfs_mean_sd = mean_sd_df(results_path='results')
dfs_mean_sd.to_parquet('data/results/dfs_mean_sd.parquet')

df_mean = mean_df(results_path='results')
df_mean.to_parquet('data/results/dfs_mean.parquet')

df_mean = sd_df(results_path='results')
df_mean.to_parquet('data/results/dfs_sd.parquet')

methods_rank_matrix = rank_matrix(df_mean)
methods_rank_matrix.to_parquet('data/results/methods_rank_matrix.parquet')

df_hyperparameters = hyperparameters_summary()
df_hyperparameters.to_parquet('data/results/df_hyperparameters.parquet')