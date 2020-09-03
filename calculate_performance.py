import os
import threading

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
            # self.memory_uses.append(current_process.memory_percent())
            self.memory_uses.append(current_process.memory_info()[0] / float(2 ** 20))
            # sleep(0.001)
            # print(current_process.cpu_percent(interval=1))

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


def main(submitid, epoch):
    load_start = datetime.now()
    vectors = load_model(model_name, epoch)
    load_finish = datetime.now()
    load_duration = load_finish - load_start
    log.info(f"model loaded in {load_duration}")

    # main process
    # get code and problem text from DB that corresponds with input submit ID
    code, problem = get(submitid)
    # preprocessing includes normalization and tokenization
    problem_processed, comments_processed, code_only = preprocess(problem, code)
    # calculate code density
    comment_line_density, comment_char_density = get_density(comments_processed, code_only)
    # check if comment has header
    header_score = calculate_header_score(comments_processed)
    # count any parts of the problem text that exist inside comment
    problem_text_in_comments_count = problem_text_in_comments(problem_processed, comments_processed)
    # remove stop words before similarity calculation
    problem_processed, comments_processed = strip_stopwords(problem_processed, comments_processed)
    # calculate average similarity score
    sim_score = calculate_similarity(vectors, problem_processed, comments_processed, 'wmd')

    # output results
    print("=================================================")
    print(f"comment density (char)                    : {comment_char_density}"
          f"\ncomment density (line)                    : {comment_line_density}"
          f"\nheader score                              : {header_score}"
          f"\nsimilarity score                          : {sim_score}"
          f"\nsnippets of problem text found in comment : {problem_text_in_comments_count}")


picks = [77827, 76977, 75666, 78512, 79955]
if __name__ == '__main__':
    assert len(sys.argv) == 3, "command line argument invalid"
    log = LogWrapper(False, sys.argv[0])
    model_name = sys.argv[1]
    e = sys.argv[2]
    log.info(f"loading {model_name} model")

    results = []
    for i in picks:
        display_cpu = MonitorCPU()
        start = datetime.now()
        display_cpu.start()
        try:
            main(i, e)
            # print("=================================================")
            # print(f"program finished in {duration}")
        finally:
            display_cpu.stop()
            finish = datetime.now()
            duration = finish - start
            # print("=================================================")
            cpu_avg = sum(display_cpu.percents) / len(display_cpu.percents)
            cpu_max = max(display_cpu.percents)
            # print(f"CPU usage percents      : {display_cpu.percents}")
            # print(f"CPU usage average       : {cpu_avg}")
            # print(f"CPU usage max           : {cpu_max}")
            memory_avg = sum(display_cpu.memory_uses) / len(display_cpu.memory_uses)
            memory_max = max(display_cpu.memory_uses)
            # print(f"Memory uses             : {display_cpu.memory_uses}")
            # print(f"Memory usage average    : {memory_avg}")
            # print(f"Memory usage max        : {memory_max}")
            results.append([cpu_avg, cpu_max, memory_avg, memory_max, duration.total_seconds()])
            del display_cpu
    df = pandas.DataFrame(results, index=picks, columns=['cpu_avg', 'cpu_max', 'memory_avg', 'memory_max', 'duration'])
    print(df)
    df.to_csv(f'results/performance-{model_name}-{e}-wmd.pandas.csv')
