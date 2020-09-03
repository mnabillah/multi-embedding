import os
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
from evaluate import SCORE_WEIGHTS

path_pearson_10 = 'results/eval-10-pearson.pandas.csv'
path_pearson_50 = 'results/eval-50-pearson.pandas.csv'
path_pearson_50_10 = 'results/eval-50-10-pearson.pandas.csv'
path_mae_10 = 'results/eval-10-mae.pandas.csv'
path_mae_50 = 'results/eval-50-mae.pandas.csv'
path_mae_50_10 = 'results/eval-50-10-mae.pandas.csv'
path_performance_w2v = 'results/performance-word2vec-50-10-wmd.pandas.csv'
path_performance_gv = 'results/performance-glove-50-wmd.pandas.csv'
path_performance_ft = 'results/performance-fasttext-50-wmd.pandas.csv'

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

# CPU average
df_performance_cpu_avg = pd.DataFrame({
    'Word2Vec': df_performance_w2v['cpu_avg'].to_numpy(),
    'GloVe': df_performance_gv['cpu_avg'].to_numpy(),
    'fastText': df_performance_ft['cpu_avg'].to_numpy(),
}, index=df_performance_w2v.index)

# CPU max
df_performance_cpu_max = pd.DataFrame({
    'Word2Vec': df_performance_w2v['cpu_max'].to_numpy(),
    'GloVe': df_performance_gv['cpu_max'].to_numpy(),
    'fastText': df_performance_ft['cpu_max'].to_numpy(),
}, index=df_performance_w2v.index)

# memory average
df_performance_memory_avg = pd.DataFrame({
    'Word2Vec': df_performance_w2v['memory_avg'].to_numpy(),
    'GloVe': df_performance_gv['memory_avg'].to_numpy(),
    'fastText': df_performance_ft['memory_avg'].to_numpy(),
}, index=df_performance_w2v.index)

# memory max
df_performance_memory_max = pd.DataFrame({
    'Word2Vec': df_performance_w2v['memory_max'].to_numpy(),
    'GloVe': df_performance_gv['memory_max'].to_numpy(),
    'fastText': df_performance_ft['memory_max'].to_numpy(),
}, index=df_performance_w2v.index)

# duration
df_performance_duration = pd.DataFrame({
    'Word2Vec': [t.second for t in
                 pd.to_datetime(df_performance_w2v['duration'], format='%H:%M:%S.%f').tolist()],
    'GloVe': [t.second for t in
              pd.to_datetime(df_performance_gv['duration'], format='%H:%M:%S.%f').tolist()],
    'fastText': [t.second for t in
                 pd.to_datetime(df_performance_ft['duration'], format='%H:%M:%S.%f').tolist()],
}, index=df_performance_w2v.index)

# save figures as png
if not os.path.exists('figures'):
    os.mkdir('figures')

df_pearson_10.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/pearson_10.png')
df_pearson_50.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/pearson_50.png')
df_pearson_w2v.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/pearson_w2v.png')
df_pearson_gv.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/pearson_gv.png')
df_pearson_ft.plot.bar(figsize=(15, 10)).get_figure().savefig(
    'figures/pearson_ft.png')
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
df_performance_cpu_avg.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_cpu_avg.png')
df_performance_memory_max.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_memory_max.png')
df_performance_duration.plot.bar(rot=0).get_figure().savefig(
    'figures/performance_duration.png')
