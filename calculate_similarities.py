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
    assert len(sys.argv) == 3, "argument invalid"

    name = sys.argv[1]
    assert name in ['word2vec', 'glove', 'fasttext'], "model name invalid"

    epoch_ver = int(sys.argv[2])
    assert epoch_ver in [10, 50], "invalid epoch version"

    log = LogWrapper(True, sys.argv[0])
    main(name, epoch_ver)
