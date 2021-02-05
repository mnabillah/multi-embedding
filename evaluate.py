import sys

import numpy as np
"""
evaluate

Description
===========
This program is used to calculate Spearman correlation and MAE of the scores.

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import pandas
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error

from db import connection
from functions import SIMILARITY_COLUMN_NAMES

HEADER_SCORE_WEIGHT = 0.0
DENSITY_SCORE_WEIGHT = 0.0
SIM_SCORE_WEIGHT = 1.0 - HEADER_SCORE_WEIGHT - DENSITY_SCORE_WEIGHT
SCORE_WEIGHTS = [(0.0, 0.0),
                 (0.0, 0.1),
                 (0.0, 0.2),
                 (0.0, 0.3),
                 (0.1, 0.0),
                 (0.1, 0.1),
                 (0.1, 0.2),
                 (0.1, 0.3),
                 (0.2, 0.0),
                 (0.2, 0.1),
                 (0.2, 0.2),
                 (0.2, 0.3),
                 (0.3, 0.0),
                 (0.3, 0.1),
                 (0.3, 0.2),
                 (0.3, 0.3)]


def get_final_score(sim_score: float, header_score: float, density_score: float) -> float:
    """
    Hitung nilai terakhir dengan bobot 80-20 (80% similarity, 20% janji).
    Args:
        sim_score (float): nilai similarity
        header_score (float): nilai header (janji)
        density_score (float): kepadatan komentar relatif terhadap source code

    Returns:
        float: Nilai sesuai pembobotan
    """
    return ((sim_score * SIM_SCORE_WEIGHT) + (header_score * HEADER_SCORE_WEIGHT) + (
            density_score * DENSITY_SCORE_WEIGHT)) * 100


if __name__ == '__main__':
    assert len(sys.argv) == 2, "command line argument invalid"
    assert int(sys.argv[1]) in [10, 50], "epoch version invalid"
    epoch = sys.argv[1]

    # HEADER_SCORE_WEIGHT = float(sys.argv[1])
    # DENSITY_SCORE_WEIGHT = float(sys.argv[2])
    # score_weights = [[HEADER_SCORE_WEIGHT, DENSITY_SCORE_WEIGHT]]

    word2vec_result_file_name = f'results/similarities-word2vec-{epoch}-wmd.csv'
    glove_result_file_name = f'results/similarities-glove-{epoch}-wmd.csv'
    fasttext_result_file_name = f'results/similarities-fasttext-{epoch}-wmd.csv'
    statistik_result_file_name = 'results/similarities-statistics.csv'

    # ambil nilai word2vec
    df = pd.read_csv(word2vec_result_file_name, header=0)
    word2vec_sims = df['sim_score'].values
    header_scores = df['header_score'].values
    density_scores = df['comment_char_density'].values

    # ambil nilai glove
    df = pd.read_csv(glove_result_file_name, header=0)
    glove_sims = df['sim_score'].values

    # ambil nilai fasttext
    df = pd.read_csv(fasttext_result_file_name, header=0)
    fasttext_sims = df['sim_score'].values

    # ambil nilai statistik
    df = pd.read_csv(statistik_result_file_name, header=0)
    statistik_1_sims = df['sim1'].values
    statistik_2_sims = df['sim2'].values

    # ambil nilai manual
    manual_scores = []
    query = 'select nilai from query_results limit 500'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    for row in result:
        manual_scores.append(row[0])
    manual_scores = np.array(manual_scores)
    del query, cursor, result

    pearson_scores = []
    spearman_scores = []
    kendall_scores = []
    mae_values = []
    for hs, ds in SCORE_WEIGHTS:
        HEADER_SCORE_WEIGHT = hs
        DENSITY_SCORE_WEIGHT = ds
        SIM_SCORE_WEIGHT = 1.0 - HEADER_SCORE_WEIGHT - DENSITY_SCORE_WEIGHT

        word2vec_final_scores = []
        glove_final_scores = []
        fasttext_final_scores = []
        statistik_1_final_scores = []
        statistik_2_final_scores = []

        for i in range(len(header_scores)):
            word2vec_final_scores.append(get_final_score(
                word2vec_sims[i], header_scores[i], density_scores[i]))
            glove_final_scores.append(get_final_score(
                glove_sims[i], header_scores[i], density_scores[i]))
            fasttext_final_scores.append(get_final_score(
                fasttext_sims[i], header_scores[i], density_scores[i]))
            statistik_1_final_scores.append(get_final_score(
                statistik_1_sims[i], header_scores[i], density_scores[i]))
            statistik_2_final_scores.append(get_final_score(
                statistik_2_sims[i], header_scores[i], density_scores[i]))

        # hitung nilai pearson correlation antara tiap nilai similarity dengan nilai manual

        # Pearson Correlation
        # word2vec_pearson, _ = pearsonr(word2vec_final_scores, manual_scores)
        # glove_pearson, _ = pearsonr(glove_final_scores, manual_scores)
        # fasttext_pearson, _ = pearsonr(fasttext_final_scores, manual_scores)
        # statistik_1_pearson, _ = pearsonr(statistik_1_final_scores, manual_scores)
        # statistik_2_pearson, _ = pearsonr(statistik_2_final_scores, manual_scores)

        # Spearman Rho Correlation
        word2vec_spearman, _ = spearmanr(
            word2vec_final_scores, manual_scores, nan_policy='propagate')
        glove_spearman, _ = spearmanr(
            glove_final_scores, manual_scores, nan_policy='propagate')
        fasttext_spearman, _ = spearmanr(
            fasttext_final_scores, manual_scores, nan_policy='propagate')
        statistik_1_spearman, _ = spearmanr(
            statistik_1_final_scores, manual_scores, nan_policy='propagate')
        statistik_2_spearman, _ = spearmanr(
            statistik_2_final_scores, manual_scores, nan_policy='propagate')

        # Kendall Tau Correlation
        word2vec_kendall, _ = kendalltau(
            word2vec_final_scores, manual_scores, nan_policy='propagate')
        glove_kendall, _ = kendalltau(
            glove_final_scores, manual_scores, nan_policy='propagate')
        fasttext_kendall, _ = kendalltau(
            fasttext_final_scores, manual_scores, nan_policy='propagate')
        statistik_1_kendall, _ = kendalltau(
            statistik_1_final_scores, manual_scores, nan_policy='propagate')
        statistik_2_kendall, _ = kendalltau(
            statistik_2_final_scores, manual_scores, nan_policy='propagate')

        # MAE
        # average of every absolute delta between real score and predicted score
        # word2vec_mae = mean_absolute_error(manual_scores, word2vec_final_scores)
        # glove_mae = mean_absolute_error(manual_scores, glove_final_scores)
        # fasttext_mae = mean_absolute_error(manual_scores, fasttext_final_scores)
        # statistik_1_mae = mean_absolute_error(manual_scores, statistik_1_final_scores)
        # statistik_2_mae = mean_absolute_error(manual_scores, statistik_2_final_scores)

        # pearson_row = [word2vec_pearson, glove_pearson, fasttext_pearson, statistik_1_pearson, statistik_2_pearson]
        spearman_row = [word2vec_spearman, glove_spearman, fasttext_spearman, statistik_1_spearman,
                        statistik_2_spearman]
        kendall_row = [word2vec_kendall, glove_kendall,
                       fasttext_kendall, statistik_1_kendall, statistik_2_kendall]
        # mae_row = [word2vec_mae, glove_mae, fasttext_mae, statistik_1_mae, statistik_2_mae]

        # pearson_scores.append(pearson_row)
        spearman_scores.append(spearman_row)
        kendall_scores.append(kendall_row)
        # mae_values.append(mae_row)

    index = pandas.MultiIndex.from_tuples(
        SCORE_WEIGHTS, names=['header', 'density'])
    columns = ['Word2Vec', 'GloVe', 'fastText', 'Statistik_1', 'Statistik_2']
    # df_pearson = pandas.DataFrame(pearson_scores, index=index, columns=columns)
    df_spearman = pandas.DataFrame(
        spearman_scores, index=index, columns=columns)
    df_kendall = pandas.DataFrame(kendall_scores, index=index, columns=columns)
    # df_mae = pandas.DataFrame(mae_values, index=index, columns=columns)
    print('Spearman')
    print(df_spearman)
    print('Kendall')
    print(df_kendall)
    # print(df_mae)

    # print(df_pearson)
    # df_pearson.to_csv(f'results/eval-{epoch}-pearson.pandas.csv')
    # df_spearman.to_csv(f'results/eval-{epoch}-spearman-nan.pandas.csv')
    # df_kendall.to_csv(f'results/eval-{epoch}-kendall-nan.pandas.csv')
    # df_mae.to_csv(f'results/eval-{epoch}-mae.pandas.csv')
