"""
calculate_score

Description
===========
This module contains the code to retrieve one submission data and calculate its score.
It makes use of the functions from the functions module.
The score is calculated based on similarity score, density score, and header score.
Each score has its weighted value, which is determined by the program arguments.

Run "python calculate_score.py -h" for more information about the arguments.

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import argparse
import logging
import mysql.connector
import os
import sys
from datetime import datetime

from db import connection
from functions import load_model, preprocess, calculate_density, calculate_header_score, problem_text_in_comments, strip_stopwords, calculate_similarity


def get(submit_id: int) -> tuple:
    """
    Get source code and problem text of one submission.
    Args:
        submit_id (int): submit ID

    Returns:
        tuple: decoded source code and problem text

    """
    query = "SELECT submission.sourcecode, tsoal.soal " \
            "FROM submission, tsoal " \
            "WHERE submission.probid=tsoal.probid " \
            f"AND submission.submitid={submit_id}"
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        sourcecode, soal = cursor.fetchone()
        sourcecode = sourcecode.decode('ASCII')
    except mysql.connector.Error as db_error:
        print(f"error from database:\n{db_error}")
        exit(1)
    except UnicodeDecodeError as decode_error:
        print(f"error decoding code blob:\n{decode_error}")
        exit(2)
    else:
        return sourcecode, soal


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(
        description="Calculate score of submission using previously-trained word embeddings.")
    parser.add_argument('model', metavar='model name', type=str, choices=['word2vec', 'glove', 'fasttext'],
                        help="specifies word embedding model (word2vec, glove, or fasttext)")
    parser.add_argument('epoch', metavar='epoch version', type=str, choices=['10', '50', '50+10'],
                        help="specifies which epoch version of the model to be used (10, 50, or 50+10 (only available for word2vec))")
    parser.add_argument('submitid', metavar='submission ID', type=str,
                        help="ID of submission to be calculated")
    parser.add_argument('density_weight', metavar='density score weight', type=float,
                        help='weight of code density score for final scoring (in decimal, e.g. 0.2)')
    parser.add_argument('header_weight', metavar='header score weight', type=float,
                        help='weight of header score for final scoring (in decimal, e.g. 0.2)')
    parser.add_argument('-v', '--verbose',
                        help="increase output verbosity", action='store_true')
    parser.add_argument('-sl', '--save_log',
                        help="saves log file into logs folder", action='store_true')
    args = parser.parse_args()

    # store each argument values
    m = args.model.lower()
    e = args.epoch
    i = args.submitid
    d = args.density_weight
    h = args.header_weight
    s = round(1.0 - d - h, 2)
    v = args.verbose
    sl = args.save_log

    # print(m)
    # print(e)
    # print(i)
    # print(d)
    # print(h)
    # print(s)
    # print(v)
    # print(sl)

    # assert that sum of density weight and header weight does not exceed 1.0 (100%)
    assert s > 0.0, "sum of density and header score weight value exceeds 1.0 (100%), must be lower"
    if e == '50+10':
        # assert model is word2vec if chosen epoch version is 50+10
        assert m == 'word2vec', "epoch version 50+10 is only available for Word2vec model"

    # configure logger
    if v:
        if sl:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            # set logging to save log file to logs folder
            logging.basicConfig(filename=f"logs/{sys.argv[0].replace('.py', '')}-{datetime.now().strftime('%Y%d%m%H%M%S')}.log",
                                filemode='w',
                                level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)
    logging.info(
        f"using {m}.{e} model to calculate submitid {i}") if v else None

    # load word embedding model
    start = datetime.now()
    vectors = load_model(m, e)
    logging.info(f"model loaded in {datetime.now() - start}") if v else None

    # get source code and problem text from database that corresponds with input submit ID
    code, problem = get(i)

    # preprocessing includes normalization and tokenization
    logging.info("preprocessing code and problem text...") if v else None
    problem_processed, comments_processed, code_only = preprocess(problem,
                                                                  code)
    # count words in code comment
    comment_word_count_raw = 0
    for line in comments_processed:
        comment_word_count_raw += len(line)
    logging.info("preprocessing finished") if v else None

    # calculate code density
    logging.info("calculating code density...") if v else None
    comment_line_density, comment_char_density = calculate_density(comments_processed,
                                                                   code_only)
    logging.info("finished calculating") if v else None

    # calculate code header score
    logging.info("calculating header score...") if v else None
    header_score = calculate_header_score(comments_processed)
    logging.info("finished calculating") if v else None

    # count any parts of the problem text that exist inside comment
    problem_text_in_comments_count = problem_text_in_comments(problem_processed,
                                                              comments_processed)

    # remove stop words before similarity calculation
    problem_processed, comments_processed = strip_stopwords(problem_processed,
                                                            comments_processed)

    # calculate average similarity score
    logging.info("calculating similarity score...") if v else None
    sim_score = calculate_similarity(vectors,
                                     problem_processed,
                                     comments_processed)
    logging.info("finished calculating") if v else None

    # change scale of scores from 0.0-1.0 to 0-100
    comment_char_density *= 100
    header_score *= 100
    sim_score *= 100

    # calculate final score based on score weights
    final_score = ((d * comment_char_density) +
                   (h * header_score) +
                   (s * sim_score))

    # count words in code comment (post-tokenization)
    comment_word_count = 0
    for line in comments_processed:
        comment_word_count += len(line)

    # output results
    print("================================================") if v else None
    print(f"comment word count (before tokenization)  : {comment_word_count_raw}\n"
          f"comment word count (after tokenization)   : {comment_word_count}\n"
          f"comment density                           : {comment_char_density}\n"
          f"header score                              : {header_score}\n"
          f"similarity score                          : {sim_score}\n"
          f"final score                               : {final_score}\n"
          f"snippets of problem text found in comment : {problem_text_in_comments_count}\n")
