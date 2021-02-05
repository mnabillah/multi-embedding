"""
calculate_performance

Description
===========
This program is used to measure performance of score calculation on five set submission data using all three models.

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import os
import threading
from datetime import datetime

import mysql.connector
import pandas
import psutil

from db import connection
from functions import *


class MonitorCPU(threading.Thread):
    """ Source: https://stackoverflow.com/a/58403731
    """

    def __init__(self) -> None:
        super().__init__()
        self.percents = []
        self.memory_uses = []

    def run(self):
        self.running = True

        current_process = psutil.Process(os.getpid())

        while self.running:
            self.percents.append(current_process.cpu_percent(interval=1))
            self.memory_uses.append(
                current_process.memory_info()[0] / float(2 ** 20))
            # sleep(0.001)

    def stop(self):
        self.running = False


def get(submit_id: int) -> tuple:
    """
    Get source code and problem text.
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


def main(submitid, epoch, model_name):
    load_start = datetime.now()
    vectors = load_model(model_name, epoch)
    load_finish = datetime.now()
    load_duration = load_finish - load_start
    # log.info(f"model loaded in {load_duration}")

    # main process
    # get code and problem text from DB that corresponds with input submit ID
    code, problem = get(submitid)
    # preprocessing includes normalization and tokenization
    problem_processed, comments_processed, code_only = preprocess(
        problem, code)
    # calculate code density
    comment_line_density, comment_char_density = calculate_density(
        comments_processed, code_only)
    # check if comment has header
    header_score = calculate_header_score(comments_processed)
    # count any parts of the problem text that exist inside comment
    problem_text_in_comments_count = problem_text_in_comments(
        problem_processed, comments_processed)
    # remove stop words before similarity calculation
    problem_processed, comments_processed = strip_stopwords(
        problem_processed, comments_processed)
    # calculate average similarity score
    sim_score = calculate_similarity(
        vectors, problem_processed, comments_processed, 'wmd')

    # output results
    print(f"comment density (char)                    : {comment_char_density}\n"
          f"comment density (line)                    : {comment_line_density}\n"
          f"header score                              : {header_score}\n"
          f"similarity score                          : {sim_score}\n"
          f"snippets of problem text found in comment : {problem_text_in_comments_count}\n")
    print("=================================================\n")


picks = [77827, 76977, 75666, 78512, 79955]
models = ['word2vec', 'glove', 'fasttext']
if __name__ == '__main__':
    assert len(sys.argv) == 2, "command line argument invalid"
    log = LogWrapper(False, sys.argv[0])
    e = sys.argv[1]

    cpu_all_model = []
    for m in models:
        # results = []
        cpu_per_model = []
        for i in picks:
            log.info(f"loading {m} model, calculating submitid {i}")
            display_cpu = MonitorCPU()
            display_cpu.start()
            start = datetime.now()
            try:
                main(i, e, m)
            finally:
                duration = datetime.now() - start
                display_cpu.stop()
                # print("=================================================")
                # print(f"program finished in {duration}")
                # print("=================================================")
                # cpu_avg = sum(display_cpu.percents) / len(display_cpu.percents)
                # cpu_max = max(display_cpu.percents)
                # print(f"CPU usage percents      : {display_cpu.percents}")
                # print(f"CPU usage average       : {cpu_avg}")
                # print(f"CPU usage max           : {cpu_max}")
                # memory_avg = sum(display_cpu.memory_uses) / \
                #     len(display_cpu.memory_uses)
                # memory_max = max(display_cpu.memory_uses)
                # print(f"Memory uses             : {display_cpu.memory_uses}")
                # print(f"Memory usage average    : {memory_avg}")
                # print(f"Memory usage max        : {memory_max}")
                # results.append([cpu_avg, cpu_max, memory_avg,
                #                 memory_max, duration.total_seconds()])
                cpu_per_model.append(display_cpu.percents)
                del display_cpu
        cpu_all_model.append(cpu_per_model)
        del cpu_per_model
    df = pandas.DataFrame(cpu_all_model, index=models, columns=picks)
    df = df.transpose()
    df.to_csv(f"results/cpu_hist-{'_'.join(models)}-{e}-wmd.pandas.csv")
    # df = pandas.DataFrame(results, index=picks, columns=[
    #                     'cpu_avg', 'cpu_max', 'memory_avg', 'memory_max', 'duration'])
    # print(df)
    # df.to_csv(f'results/performance-{model_name}-{e}-wmd.pandas.csv')
