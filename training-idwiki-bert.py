import logging
import os
import sys
from datetime import datetime

from transformers import AutoTokenizer, AutoModelWithLMHead

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(
        filename="logs\\{}-{}.log".format(program.replace('.py', ''), datetime.now().strftime('%Y-%d-%m')),
        filemode='w',
        format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
        datefmt='%d-%b-%y %H:%M:%S')

    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
    model = AutoModelWithLMHead.from_pretrained("cahya/bert-base-indonesian-522M")
