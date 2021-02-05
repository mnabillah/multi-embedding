"""
calculate_all

Description
===========
This program is used for experiment phase to calculate header, density, and similarity score for all data.
It makes use of the functions from the functions module.
The output of this program is a csv file in the results folder.

Run "python calculate_all.py -h" for more information about the arguments.

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import argparse
import csv
import os

from functions import *


def main(model_name: str, epoch: int):
    """Main method.

    Args:
        model_name (str): determines the word embedding model
        epoch (int): determines which version of the model (10 or 50 epochs)
    """
    # fetch data
    results = get_data()

    # load model
    log.info(f"loading {model_name} model...")
    start = datetime.now()
    vectors = load_model(model_name, epoch)
    finish = datetime.now()
    duration = finish - start
    log.info(f"model loaded in {duration}")

    # start measuring similarity for every data
    log.info(f'running similarity calculation with {model_name} model...')
    start = datetime.now()
    scores = get_sim_scores(results, vectors)
    finish = datetime.now()
    duration = finish - start
    log.info(f"similarity calculation finished in: {duration}")
    log.info(f"average calculation time per code {duration / 500}")

    # save to csv writer
    log.info(f"saving to CSV...")
    if not os.path.exists('results'):
        os.mkdir('results')
    csv_file = open(f'results/similarities-{model_name.lower()}-{epoch}-wmd.csv',
                    'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(SIMILARITY_COLUMN_NAMES)
    for line in scores:
        writer.writerow([line[0], line[1], line[2], line[3], line[4], line[5]])


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        description="Calculate score of submission using previously-trained word embeddings.")
    parser.add_argument('model', metavar='model name', type=str, choices=['word2vec', 'glove', 'fasttext'],
                        help="specifies word embedding model (word2vec, glove, or fasttext)")
    parser.add_argument('epoch', metavar='epoch version', type=str, choices=['10', '50', '50+10'],
                        help="specifies which epoch version of the model to be used (10, 50, or 50+10 (only available for word2vec))")
    args = parser.parse_args()

    name = args.model.lower()
    epoch_ver = args.epoch

    log = LogWrapper(True, sys.argv[0])
    main(name, epoch_ver)
