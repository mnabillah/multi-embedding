import sys

import numpy
import pandas
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from db import connection
from functions import SIMILARITY_COLUMN_NAMES

# TODO: pastikan bobot nilai
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

    word2vec_sims = []
    glove_sims = []
    fasttext_sims = []
    statistik_1_sims = []
    statistik_2_sims = []

    header_scores = []
    density_scores = []

    manual_scores = []

    word2vec_result_file_name = f'results/similarities-word2vec-{epoch}-wmd.csv'
    glove_result_file_name = f'results/similarities-glove-{epoch}-wmd.csv'
    fasttext_result_file_name = f'results/similarities-fasttext-{epoch}-wmd.csv'
    statistik_result_file_name = 'results/similarities-statistics.csv'

    # ambil nilai word2vec
    df = pd.read_csv(word2vec_result_file_name, names=SIMILARITY_COLUMN_NAMES, header=0)
    for index, row in df.iterrows():
        word2vec_sims.append(row['sim_score'])
        header_scores.append(row['header_score'])
        density_scores.append(row['comment_char_density'])
    del df

    # ambil nilai glove
    df = pd.read_csv(glove_result_file_name, names=SIMILARITY_COLUMN_NAMES, header=0)
    for index, row in df.iterrows():
        glove_sims.append(row['sim_score'])
    del df

    # ambil nilai fasttext
    df = pd.read_csv(fasttext_result_file_name, names=SIMILARITY_COLUMN_NAMES, header=0)
    for index, row in df.iterrows():
        fasttext_sims.append(row['sim_score'])
    del df

    # ambil nilai statistik
    df = pd.read_csv(statistik_result_file_name, names=['header_score', 'sim1', 'sim1nanotime', 'sim2', 'sim2nanotime'],
                     header=0)
    for index, row in df.iterrows():
        sim1 = row['sim1']
        sim2 = row['sim2']
        statistik_1_sims.append(sim1 if not numpy.isnan(sim1) and not numpy.isinf(sim2) else 0)
        statistik_2_sims.append(sim2 if not numpy.isnan(sim2) and not numpy.isinf(sim2) else 0)
    del df

    # ambil nilai manual
    query = 'select nilai from query_results limit 500'
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    for row in result:
        manual_scores.append(row[0])
    del query, cursor, result

    pearson_scores = []
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
            word2vec_final_scores.append(get_final_score(word2vec_sims[i], header_scores[i], density_scores[i]))
            glove_final_scores.append(get_final_score(glove_sims[i], header_scores[i], density_scores[i]))
            fasttext_final_scores.append(get_final_score(fasttext_sims[i], header_scores[i], density_scores[i]))
            statistik_1_final_scores.append(get_final_score(statistik_1_sims[i], header_scores[i], density_scores[i]))
            statistik_2_final_scores.append(get_final_score(statistik_2_sims[i], header_scores[i], density_scores[i]))

        # hitung nilai pearson correlation antara tiap nilai similarity dengan nilai manual

        # Pearson Correlation
        word2vec_pearson, _ = pearsonr(word2vec_final_scores, manual_scores)
        glove_pearson, _ = pearsonr(glove_final_scores, manual_scores)
        fasttext_pearson, _ = pearsonr(fasttext_final_scores, manual_scores)
        statistik_1_pearson, _ = pearsonr(statistik_1_final_scores, manual_scores)
        statistik_2_pearson, _ = pearsonr(statistik_2_final_scores, manual_scores)

        # MAE
        # average of every absolute delta between real score and predicted score
        word2vec_mae = mean_absolute_error(manual_scores, word2vec_final_scores)
        glove_mae = mean_absolute_error(manual_scores, glove_final_scores)
        fasttext_mae = mean_absolute_error(manual_scores, fasttext_final_scores)
        statistik_1_mae = mean_absolute_error(manual_scores, statistik_1_final_scores)
        statistik_2_mae = mean_absolute_error(manual_scores, statistik_2_final_scores)

        pearson_row = [word2vec_pearson, glove_pearson, fasttext_pearson, statistik_1_pearson, statistik_2_pearson]
        mae_row = [word2vec_mae, glove_mae, fasttext_mae, statistik_1_mae, statistik_2_mae]

        pearson_scores.append(pearson_row)
        mae_values.append(mae_row)

    index = pandas.MultiIndex.from_tuples(SCORE_WEIGHTS, names=['header', 'density'])
    columns = ['Word2Vec', 'GloVe', 'fastText', 'Statistik_1', 'Statistik_2']
    df_pearson = pandas.DataFrame(pearson_scores, index=index, columns=columns)
    df_mae = pandas.DataFrame(mae_values, index=index, columns=columns)

    print(df_pearson)
    df_pearson.to_csv(f'results/eval-{epoch}-pearson.pandas.csv')
    df_mae.to_csv(f'results/eval-{epoch}-mae.pandas.csv')
