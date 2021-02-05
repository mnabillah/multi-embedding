"""
functions module

Description
===========
This module holds the functions to be used by other modules.
It includes functions to:
    1) retrieve submission data
    2) preprocess problem text and comments
    3) load word embedding model
    4) calculate similarity between code comments and problem text, and
    5) calculate code comment density and code header score

Made by:
    Muhammad Nabillah Fihira Rischa
    abel.rischa@gmail.com
"""
import logging
import re
import string
import sys
from datetime import datetime

from gensim.similarities import WmdSimilarity
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

# header template text
# Saya mengerjakan evaluasi dalam mata kuliah untuk keberkahanNya
# maka saya tidak melakukan kecurangan seperti yang telah dispesifikasikan. Aamiin
header = "Saya mengerjakan evaluasi dalam mata kuliah untuk keberkahanNya" \
         " maka saya tidak melakukan kecurangan seperti yang telah dispesifikasikan. Aamiin."
# lowercase and stripped of punctuations
header_parsed = [''.join([letter for letter
                          in word.lower()
                          if letter not in string.punctuation])
                 for word in header.split(' ')]
SIMILARITY_COLUMN_NAMES = ['row_id',
                           'comment_char_density',
                           'comment_line_density',
                           'header_score',
                           'sim_score',
                           'problem_text_in_comments_count']


class LogWrapper:
    def __init__(self, verbose, script_name):
        self.verbose = verbose
        if self.verbose:
            self.log = logging.getLogger(script_name)
            logging.basicConfig(
                filename=f"logs/{script_name.replace('.py', '')}"
                "-{datetime.now().strftime('%Y%d%m%H%M%S')}.log",
                filemode='w',
                format="%(asctime)s:\t[%(levelname)s]\t%(message)s",
                datefmt='%d-%b-%y %H:%M:%S')
            self.log.addHandler(logging.StreamHandler())
            logging.root.setLevel(level=logging.INFO)
            self.log.info("running %s" % ' '.join(sys.argv))

    def info(self, text):
        self.log.info(text) if self.verbose else print(text)

    def warning(self, text):
        self.log.warning(text) if self.verbose else print(text)

    def error(self, text):
        if self.verbose:
            self.log.error(text)
        else:
            print(text)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate levenshtein distance from two words.
    Source: https://stackoverflow.com/a/32558749

    Args:
        s1 (str): second string
        s2 (str): first string

    Returns:
        int: levenshtein distance of the input strings
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def calculate_header_score(comment_lines: list) -> float:
    """Calculate and determine the presence of a header line inside the comments.

    Args:
        comment_lines (list): list of tokenized comment lines from one source code

    Returns:
        float: value of [0, 1.0]
    """
    # create vector of 0s equal to the amount of words in our header template text
    header_word_number = [0] * len(header_parsed)
    i = 0
    # loop through header words
    for header_word in header_parsed:
        # loop through comment lines
        for line in comment_lines:
            # loop through words in a line
            for comment_word in line:
                # check if word from header equals word from comment
                if header_word == comment_word:
                    # if word from comment checks out, mark the 0 from the vector as 1
                    #   indicating that the word exists in the comments from the code
                    header_word_number[i] = 1
        i += 1

    return sum(header_word_number) / len(header_word_number)


def preprocess(problem: str, code: str):
    """Normalize and tokenize raw problem text and extract normalized comment lines from code file.
    Tokenized using RegexpTokenizer from NLTK.

    Args:
        problem (str): raw problem text from db
        code (str): raw code file, pre-encoded

    Returns:
        tuple: list of tokenized lines of problem text, list of tokenized lines of comments
    """
    tokenizer = RegexpTokenizer(r'\w+')

    # normalize problem text
    # convert to printable representation to reveal carriage return characters
    problem = repr(problem)
    # remove html tags
    problem = re.sub(r'<.*?>', '', problem)
    # remove carriage return characters
    problem = re.sub(r'\\r', ' ', problem)
    # revert from printable representation
    problem = eval(problem)
    problem = problem.split('.')
    problem = [tokenizer.tokenize(line.lower()) for line in problem]

    # trim newline and tab characters
    code = re.sub(r'[\r|\n|\t]+', '\n', code)
    # trim multiple spaces (and to handle codes with leading space as indentation)
    #   this, however, turns all multiple whitespace into one, so we need to strip the remaining leading whitespace afterwards
    code = re.sub(r' +', ' ', code)
    # remove remaining leading whitespace
    code = '\n'.join([line.strip() for line in code.split('\n')])

    # extract comments from code using regex pattern
    # regex pattern from https://www.regexpal.com/94246
    re_pattern = r'/\*[\s\S]*?\*/|([^:]|^)//.*$'
    matcher = re.compile(re_pattern, re.MULTILINE)
    matches = matcher.finditer(code)
    # iterate through found matches while removing newline and leading-trailing whitespaces
    comments = []
    for match in matches:
        comment_line = match.group()
        comments.append(tokenizer.tokenize(comment_line.lower()))

    # get only the codes by removing comments
    code_only = matcher.sub("", code).split('\n')

    return problem, comments, code_only


def problem_text_in_comments(problem: list, comments: list) -> int:
    """Method to count how much lines from the problem text exists inside the comment lines.
    This is to mark codes with higher similarity score because the student pasted a part of the problem text into the code as comments.

    Args:
        problem (list): list of tokenized problem text lines
        comments (list): list of tokenized comment lines

    Returns:
        int: amount of lines from problem text found inside the comments
    """
    comments = ' '.join([' '.join(line) for line in comments])
    count = 0
    for line in problem:
        if len(line) > 3 and ' '.join(line) in comments:
            count += 1
    return count


def get_data(evaluation_data: bool = False) -> list:
    """
    Establish connection and fetch data
    Args:
        evaluation_data (bool): if True, fetch evaluation data from db_cspc

    Returns:
        list: data from database
    """
    import db

    conn = db.connection
    if evaluation_data:
        # query = 'select s.login, true, tsoal.judul, tsoal.soal, tsoal.input, tsoal.output, tsoal.input2, tsoal.output2, tsoal.input3, tsoal.output3, tsoal.pembuat, s.sourcecode FROM (select login, probid, sourcecode, headercode, submitid from submission where ((submittime like \'2019-%\') or (submittime like \'2018-%\') or (submittime like \'2017-%\')) and (sourcecode like \'%/*%\' or sourcecode like \'%//%\') AND headercode IS NULL ORDER BY submittime DESC LIMIT 0,2000) AS s, tsoal, judging, judging_run WHERE s.probid=tsoal.probid AND s.submitid=judging.submitid AND judging.judgingid=judging_run.judgingid AND  judging_run.output_error LIKE \'Correct%\' AND s.probid=tsoal.probid LIMIT 500'
        query = 'select true, tsoal.soal, s.sourcecode, true FROM (select login, probid, sourcecode, headercode, submitid from submission where ((submittime like \'2019-%\') or (submittime like \'2018-%\') or (submittime like \'2017-%\')) and (sourcecode like \'%/*%\' or sourcecode like \'%//%\') AND headercode IS NULL ORDER BY submittime DESC LIMIT 0,2000) AS s, tsoal, judging, judging_run WHERE s.probid=tsoal.probid AND s.submitid=judging.submitid AND judging.judgingid=judging_run.judgingid AND  judging_run.output_error LIKE \'Correct%\' AND s.probid=tsoal.probid LIMIT 0,500'
    else:
        query = 'select id, soal, sourcecode, nilai from query_results limit 0,500'

    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def get_all_soal():
    import db
    conn = db.connection
    query = "select soal from tsoal"
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def preprocess_soal(problem):
    tokenizer = RegexpTokenizer(r'\w+')

    # normalize problem text
    # convert to printable representation to reveal carriage return characters
    problem = repr(problem)
    # remove html tags
    problem = re.sub(r'<.*?>', '', problem)
    # remove carriage return characters
    problem = re.sub(r'\\r', ' ', problem)
    # revert from printable representation
    problem = eval(problem)
    problem = problem.split('.')
    problem = [tokenizer.tokenize(line.lower()) for line in problem]
    return problem


def strip_stopwords(problem, comments):
    problem = [[word for word in line if word not in stopwords.words(
        'indonesian')] for line in problem]
    comments = [[word for word in line if word not in stopwords.words(
        'indonesian')] for line in comments]
    return problem, comments


def calculate_density(comments, code):
    # calculate code density
    comment_line_count = len(comments)
    code_line_count = len(code)
    comment_char_count = len(''.join([''.join(line) for line in comments]))
    code_char_count = len(''.join(code).strip(' '))
    comment_line_density = comment_line_count / \
        (comment_line_count + code_line_count)
    comment_char_density = comment_char_count / \
        (comment_char_count + code_char_count)
    return comment_line_density, comment_char_density


def get_sim_scores(data, vectors):
    """
    Get all similarity score of every data row
    Args:
        data (list): data from database
        vectors: word vectors

    Returns:
        list: list of similarity score and other data from the source code

    """
    results = []
    for row_id, problem, code, _ in data:
        # in case of decoding error
        try:
            # decode source code
            code = code.decode('ASCII')
        except UnicodeDecodeError:
            logging.error(f'error decoding blob at row={row_id}')
        else:
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
            print(f"calculating similarity for row {row_id}", end='\r')
            sim_score = calculate_similarity(
                vectors, problem_processed, comments_processed)
            results.append([row_id,
                            comment_char_density,
                            comment_line_density,
                            header_score,
                            sim_score if sim_score is not None else '-',
                            problem_text_in_comments_count])
    return results


def load_model(model_name, epoch):
    from gensim.models import KeyedVectors
    from gensim.models.keyedvectors import FastTextKeyedVectors, Word2VecKeyedVectors
    from gensim.models.fasttext import load_facebook_vectors, load_facebook_model
    from gensim.models.wrappers import FastText

    if epoch != '50+10':
        # if epoch choice is epoch 10 or 50 (no continued training with CSPC problem texts)
        if model_name.lower() == 'word2vec':
            return Word2VecKeyedVectors.load(f"trained_models/word2vec/idwiki.epoch-{epoch}.dim-300.kv")
        elif model_name.lower() == 'glove':
            return KeyedVectors.load_word2vec_format(
                f"trained_models/glove/converted.idwiki.epoch-{epoch}.dim-300.model.txt")
        elif model_name.lower() == 'fasttext':
            model = FastText.load_fasttext_format(
                f"trained_models/fasttext/idwiki.epoch-{epoch}.dim-300.bin")
            return model.wv
    else:
        # if epoch choice is 50+10, i.e. the 50 epoch word2vec model that's trained further with CSPC problem texts
        return Word2VecKeyedVectors.load(f"trained_models/word2vec/idwiki-cspc.epoch-50.dim-300.kv")

    return None


def calculate_similarity(vectors, problem_text, comments):
    """
    Calculate similarity score.
    Args:
        vectors: word vectors trained by Word2Vec, GloVe, or fastText
        problem_text: list of tokens from the problem text
        comments: list of tokens from code comments

    Returns:
        float: similarity score
    """
    if vectors is not None:
        wmd_index = WmdSimilarity(problem_text, vectors)
        wmd = wmd_index[comments]
        wmd = sum(wmd) / len(wmd)
        import numpy as np
        if type(wmd) is not float and type(wmd) is not np.float64:
            wmd = sum(wmd) / len(wmd)
        return wmd

    return None
