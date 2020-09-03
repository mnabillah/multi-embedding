import csv
from datetime import datetime

w2v = f"eval-wmd-word2vec-{datetime.now().year}-{datetime.now().month:02d}-{datetime.now().day:02d}.csv"
glove = f"eval-wmd-glove-{datetime.now().year}-{datetime.now().month:02d}-{datetime.now().day:02d}.csv"
fasttext = f"eval-wmd-fasttext-{datetime.now().year}-{datetime.now().month:02d}-{datetime.now().day:02d}.csv"

row_ids = []
line_counts = []
word_counts = []
header_scores = []
w2v_scores = []
glove_scores = []
fasttext_scores = []
problem_text_in_comment_counts = []

# get fasttext scores
with open(fasttext, 'r', newline='', encoding='utf8') as fasttext_csv:
    fasttext_reader = csv.reader(fasttext_csv)
    next(fasttext_reader)
    for line in fasttext_reader:
        fasttext_scores.append(line[4])

# get glove scores
with open(glove, 'r', newline='', encoding='utf8') as glove_csv:
    glove_reader = csv.reader(glove_csv)
    next(glove_reader)
    for line in glove_reader:
        glove_scores.append(line[4])

# get word2vec scores and metadata
with open(w2v, 'r', newline='', encoding='utf8') as w2v_csv:
    w2v_reader = csv.reader(w2v_csv)
    next(w2v_reader)
    for line in w2v_reader:
        row_ids.append(line[0])
        line_counts.append(line[1])
        word_counts.append(line[2])
        header_scores.append(line[3])
        w2v_scores.append(line[4])
        problem_text_in_comment_counts.append(line[5])

combined_list = zip(row_ids, line_counts, word_counts, header_scores, w2v_scores, glove_scores,
                    fasttext_scores, problem_text_in_comment_counts)

combined = f"eval-wmd-combined-{datetime.now().year}-{datetime.now().month:02d}-{datetime.now().day:02d}.csv"
with open(combined, 'w', newline='', encoding='utf8') as combine_csv:
    writer = csv.writer(combine_csv)
    writer.writerow(
        ['id', 'comment_line_count', 'comment_word_count', 'header_score', 'w2v_score', 'glove_score', 'fasttext_score',
         'problem_text_in_comment_counts'])
    for line in combined_list:
        writer.writerow(line)
