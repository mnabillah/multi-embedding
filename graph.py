import os
from ast import literal_eval
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
from evaluate import SCORE_WEIGHTS

path_pearson_10 = 'results/eval-10-pearson.pandas.csv'
path_pearson_50 = 'results/eval-50-pearson.pandas.csv'
path_pearson_50_10 = 'results/eval-50-10-pearson.pandas.csv'
path_spearman_10 = 'results/eval-10-spearman.pandas.csv'
path_spearman_50 = 'results/eval-50-spearman.pandas.csv'
path_spearman_50_10 = 'results/eval-50-10-spearman.pandas.csv'
path_kendall_10 = 'results/eval-10-kendall-nan.pandas.csv'
path_kendall_50 = 'results/eval-50-kendall-nan.pandas.csv'
path_kendall_50_10 = 'results/eval-50-10-kendall.pandas.csv'
path_mae_10 = 'results/eval-10-mae.pandas.csv'
path_mae_50 = 'results/eval-50-mae.pandas.csv'
path_mae_50_10 = 'results/eval-50-10-mae.pandas.csv'
path_performance_w2v = 'results/performance-word2vec-50-10-wmd.pandas.csv'
path_performance_gv = 'results/performance-glove-50-wmd.pandas.csv'
path_performance_ft = 'results/performance-fasttext-50-wmd.pandas.csv'
path_performance_sim1 = 'results/performance-sim1.csv'
path_performance_sim2 = 'results/performance-sim2.csv'
path_performance_lcs = 'results/performance-lcs.csv'
path_cpu_hist_10 = 'results/cpu_hist-word2vec_glove_fasttext-10-wmd.pandas.csv'
path_cpu_hist_50 = 'results/cpu_hist-word2vec_glove_fasttext-50-wmd.pandas.csv'

# Pearson
# 10 epoch
df_pearson_10 = pd.read_csv(path_pearson_10, header=0, index_col=[0, 1])
# 50 epoch
df_pearson_50 = pd.read_csv(path_pearson_50, header=0, index_col=[0, 1])
# 50 epoch (+10 epoch CSPC)
df_pearson_50_10 = pd.read_csv(path_pearson_50_10, header=0, index_col=[0, 1])
# per model
df_pearson_w2v = pd.DataFrame({
    '10': df_pearson_10['Word2Vec'].to_numpy(),
    '50': df_pearson_50['Word2Vec'].to_numpy(),
    '50+10': df_pearson_50_10['Word2Vec'].to_numpy(),
}, index=df_pearson_10.index)
df_pearson_gv = pd.DataFrame({
    '10': df_pearson_10['GloVe'].to_numpy(),
    '50': df_pearson_50['GloVe'].to_numpy(),
}, index=df_pearson_10.index)
df_pearson_ft = pd.DataFrame({
    '10': df_pearson_10['fastText'].to_numpy(),
    '50': df_pearson_50['fastText'].to_numpy(),
}, index=df_pearson_10.index)
df_pearson_statistik = pd.DataFrame({
    'Statistik_1': df_pearson_10['Statistik_1'].to_numpy(),
    'Statistik_2': df_pearson_10['Statistik_2'].to_numpy(),
}, index=df_pearson_10.index)

# Spearman
# 10 epoch
df_spearman_10 = pd.read_csv(path_spearman_10, header=0, index_col=[0, 1])
# 50 epoch
df_spearman_50 = pd.read_csv(path_spearman_50, header=0, index_col=[0, 1])
# 50 epoch (+10 epoch CSPC)
df_spearman_50_10 = pd.read_csv(
    path_spearman_50_10, header=0, index_col=[0, 1])
# per model
df_spearman_w2v = pd.DataFrame({
    '10': df_spearman_10['Word2Vec'].to_numpy(),
    '50': df_spearman_50['Word2Vec'].to_numpy(),
    '50+10': df_spearman_50_10['Word2Vec'].to_numpy(),
}, index=df_spearman_10.index)
df_spearman_gv = pd.DataFrame({
    '10': df_spearman_10['GloVe'].to_numpy(),
    '50': df_spearman_50['GloVe'].to_numpy(),
}, index=df_spearman_10.index)
df_spearman_ft = pd.DataFrame({
    '10': df_spearman_10['fastText'].to_numpy(),
    '50': df_spearman_50['fastText'].to_numpy(),
}, index=df_spearman_10.index)
df_spearman_statistik = pd.DataFrame({
    'Statistik_1': df_spearman_10['Statistik_1'].to_numpy(),
    'Statistik_2': df_spearman_10['Statistik_2'].to_numpy(),
}, index=df_spearman_10.index)

# Kendall
# 10 epoch
df_kendall_10 = pd.read_csv(path_kendall_10, header=0, index_col=[0, 1])
# 50 epoch
df_kendall_50 = pd.read_csv(path_kendall_50, header=0, index_col=[0, 1])
# 50 epoch (+10 epoch CSPC)
df_kendall_50_10 = pd.read_csv(path_kendall_50_10, header=0, index_col=[0, 1])
# per model
df_kendall_w2v = pd.DataFrame({
    '10': df_kendall_10['Word2Vec'].to_numpy(),
    '50': df_kendall_50['Word2Vec'].to_numpy(),
    '50+10': df_kendall_50_10['Word2Vec'].to_numpy(),
}, index=df_kendall_10.index)
df_kendall_gv = pd.DataFrame({
    '10': df_kendall_10['GloVe'].to_numpy(),
    '50': df_kendall_50['GloVe'].to_numpy(),
}, index=df_kendall_10.index)
df_kendall_ft = pd.DataFrame({
    '10': df_kendall_10['fastText'].to_numpy(),
    '50': df_kendall_50['fastText'].to_numpy(),
}, index=df_kendall_10.index)
df_kendall_statistik = pd.DataFrame({
    'Statistik_1': df_kendall_10['Statistik_1'].to_numpy(),
    'Statistik_2': df_kendall_10['Statistik_2'].to_numpy(),
}, index=df_kendall_10.index)

# MAE
# 10 epoch
df_mae_10 = pd.read_csv(path_mae_10, header=0, index_col=[0, 1])
# 50 epoch
df_mae_50 = pd.read_csv(path_mae_50, header=0, index_col=[0, 1])
# 50 epoch (+10 epoch CSPC)
df_mae_50_10 = pd.read_csv(path_mae_50_10, header=0, index_col=[0, 1])
# per model
df_mae_w2v = pd.DataFrame({
    '10': df_mae_10['Word2Vec'].to_numpy(),
    '50': df_mae_50['Word2Vec'].to_numpy(),
    '50+10': df_mae_50_10['Word2Vec'].to_numpy(),
}, index=df_mae_10.index)
df_mae_gv = pd.DataFrame({
    '10': df_mae_10['GloVe'].to_numpy(),
    '50': df_mae_50['GloVe'].to_numpy(),
}, index=df_mae_10.index)
df_mae_ft = pd.DataFrame({
    '10': df_mae_10['fastText'].to_numpy(),
    '50': df_mae_50['fastText'].to_numpy(),
}, index=df_mae_10.index)
df_mae_statistik = pd.DataFrame({
    'Statistik_1': df_mae_10['Statistik_1'].to_numpy(),
    'Statistik_2': df_mae_10['Statistik_2'].to_numpy(),
}, index=df_mae_10.index)

# Performance
# Word2Vec
df_performance_w2v = pd.read_csv(path_performance_w2v, header=0, index_col=0)
# GloVe
df_performance_gv = pd.read_csv(path_performance_gv, header=0, index_col=0)
# fastText
df_performance_ft = pd.read_csv(path_performance_ft, header=0, index_col=0)
# sim1
df_performance_sim1 = pd.read_csv(path_performance_sim1, header=0, index_col=0)
# sim2
df_performance_sim2 = pd.read_csv(path_performance_sim2, header=0, index_col=0)
# lcs
df_performance_lcs = pd.read_csv(path_performance_lcs, header=0, index_col=0)
# CPU average
df_performance_cpu_avg = pd.DataFrame({
    'Word2Vec': df_performance_w2v['cpu_avg'].to_numpy(),
    'GloVe': df_performance_gv['cpu_avg'].to_numpy(),
    'fastText': df_performance_ft['cpu_avg'].to_numpy(),
    'Statistik 1': df_performance_sim1['cpu'].to_numpy(),
    'Statistik 2': df_performance_sim2['cpu'].to_numpy(),
    'LCS': df_performance_lcs['cpu'].to_numpy(),
}, index=df_performance_w2v.index)
# CPU max
df_performance_cpu_max = pd.DataFrame({
    'Word2Vec': df_performance_w2v['cpu_max'].to_numpy(),
    'GloVe': df_performance_gv['cpu_max'].to_numpy(),
    'fastText': df_performance_ft['cpu_max'].to_numpy(),
    'Statistik 1': df_performance_sim1['cpu'].to_numpy(),
    'Statistik 2': df_performance_sim2['cpu'].to_numpy(),
    'LCS': df_performance_lcs['cpu'].to_numpy(),
}, index=df_performance_w2v.index)
# memory average
df_performance_memory_avg = pd.DataFrame({
    'Word2Vec': df_performance_w2v['memory_avg'].to_numpy(),
    'GloVe': df_performance_gv['memory_avg'].to_numpy(),
    'fastText': df_performance_ft['memory_avg'].to_numpy(),
    'Statistik 1': df_performance_sim1['memory'].to_numpy(),
    'Statistik 2': df_performance_sim2['memory'].to_numpy(),
    'LCS': df_performance_lcs['memory'].to_numpy(),
}, index=df_performance_w2v.index)
# memory max
df_performance_memory_max = pd.DataFrame({
    'Word2Vec': df_performance_w2v['memory_max'].to_numpy(),
    'GloVe': df_performance_gv['memory_max'].to_numpy(),
    'fastText': df_performance_ft['memory_max'].to_numpy(),
    'Statistik 1': df_performance_sim1['memory'].to_numpy(),
    'Statistik 2': df_performance_sim2['memory'].to_numpy(),
    'LCS': df_performance_lcs['memory'].to_numpy(),
}, index=df_performance_w2v.index)
# duration
df_performance_duration = pd.DataFrame({
    'Word2Vec': pd.to_timedelta(df_performance_w2v['duration']).dt.total_seconds(),
    'GloVe': pd.to_timedelta(df_performance_gv['duration']).dt.total_seconds(),
    'fastText': pd.to_timedelta(df_performance_ft['duration']).dt.total_seconds(),
    'Statistik 1': df_performance_sim1['duration'].to_numpy(),
    'Statistik 2': df_performance_sim2['duration'].to_numpy(),
    'LCS': df_performance_lcs['duration'].to_numpy(),
}, index=df_performance_w2v.index)
df_performance_cpu_avg.to_csv('results/performance_cpu_avg.pandas.csv')
df_performance_memory_max.to_csv('results/performance_memory_avg.pandas.csv')
df_performance_duration.to_csv('results/performance_duration.pandas.csv')

# save figures as png
if not os.path.exists('figures'):
    os.mkdir('figures')

df_pearson_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_10.png')
df_pearson_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_50.png')
df_pearson_w2v.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_w2v.png')
df_pearson_gv.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_gv.png')
df_pearson_ft.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_ft.png')
df_spearman_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_10.png')
df_spearman_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_50.png')
df_spearman_w2v.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_w2v.png')
df_spearman_gv.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_gv.png')
df_spearman_ft.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_ft.png')
df_spearman_statistik.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_statistik.png')
df_kendall_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_10.png')
df_kendall_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_50.png')
df_kendall_w2v.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_w2v.png')
df_kendall_gv.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_gv.png')
df_kendall_ft.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_ft.png')
df_mae_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_10.png')
df_mae_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_50.png')
df_mae_w2v.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_w2v.png')
df_mae_gv.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_gv.png')
df_mae_ft.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_ft.png')
df_mae_statistik.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_statistik.png')
df_performance_cpu_avg.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_cpu_avg.png')
df_performance_memory_max.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_memory_max.png')
df_performance_duration.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_duration.png')

df_pearson_embedding_only_10 = df_pearson_10.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_pearson_embedding_only_50 = df_pearson_50.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_spearman_embedding_only_10 = df_spearman_10.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_spearman_embedding_only_50 = df_spearman_50.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_kendall_embedding_only_10 = df_kendall_10.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_kendall_embedding_only_50 = df_kendall_50.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_mae_embedding_only_10 = df_mae_10.drop(
    columns=['Statistik_1', 'Statistik_2'])
df_mae_embedding_only_50 = df_mae_50.drop(
    columns=['Statistik_1', 'Statistik_2'])

df_pearson_embedding_only_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_10_embedding_only.png')
df_pearson_embedding_only_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_pearson_50_embedding_only.png')
df_spearman_embedding_only_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_10_embedding_only.png')
df_spearman_embedding_only_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_spearman_50_embedding_only.png')
df_kendall_embedding_only_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_10_embedding_only.png')
df_kendall_embedding_only_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/correlation_kendall_50_embedding_only.png')
df_mae_embedding_only_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_10_embedding_only.png')
df_mae_embedding_only_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/mae_50_embedding_only.png')
exit(0)

# CPU histogram
# credits: https://stackoverflow.com/questions/40256820/plotting-a-column-containing-lists-using-pandas
# TODO: figure out how to save
converter = {'word2vec': literal_eval,
             'glove': literal_eval, 'fasttext': literal_eval}
df_cpu_hist_10 = pd.read_csv(
    path_cpu_hist_10, header=0, index_col=0, converters=converter)
df_cpu_hist_10_w2v = df_cpu_hist_10[[
    'word2vec']].unstack().apply(pd.Series).transpose()
df_cpu_hist_10_gv = df_cpu_hist_10[['glove']
                                   ].unstack().apply(pd.Series).transpose()
df_cpu_hist_10_ft = df_cpu_hist_10[['fasttext']
                                   ].unstack().apply(pd.Series).transpose()
# credits: https://stackoverflow.com/questions/22483588/how-can-i-plot-separate-pandas-dataframes-as-subplots
fig, axes = plt.subplots(nrows=3, ncols=1)
df_cpu_hist_10_w2v.plot.line(ax=axes[0])
df_cpu_hist_10_gv.plot.line(ax=axes[1])
df_cpu_hist_10_ft.plot.line(ax=axes[2])
plt.show()
df_cpu_hist_50 = pd.read_csv(
    path_cpu_hist_50, header=0, index_col=0, converters=converter)
df_cpu_hist_50_w2v = df_cpu_hist_50[[
    'word2vec']].unstack().apply(pd.Series).transpose()
df_cpu_hist_50_gv = df_cpu_hist_50[['glove']
                                   ].unstack().apply(pd.Series).transpose()
df_cpu_hist_50_ft = df_cpu_hist_50[['fasttext']
                                   ].unstack().apply(pd.Series).transpose()
