import string
from pprint import pprint

from nltk.corpus import stopwords

from calculate_similarities import preprocess
from db import connection


def levenshtein_distance(s1: str, s2: str):
    """Calculate levenshtein distance from two words.
    Source: https://stackoverflow.com/a/32558749

    Args:
        s1 (str): second string
        s2 (str): first string

    Returns:
        float: levenshtein distance of the input strings
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


if __name__ == '__main__':
    # fetch data
    query = 'select id, soal, sourcecode, nilai from query_results limit 500'
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()

    # header template text
    # Saya mengerjakan evaluasi dalam mata kuliah untuk keberkahanNya
    # maka saya tidak melakukan kecurangan seperti yang telah dispesifikasikan. Aamiin
    header = "Saya mengerjakan evaluasi dalam mata kuliah untuk keberkahanNya" + \
        " maka saya tidak melakukan kecurangan seperti yang telah dispesifikasikan. Aamiin."
    print(header)

    # lowercase and stripped of punctuations
    header_parsed = [''.join([letter for letter in word.lower() if letter not in string.punctuation])
                     for word in header.split(' ')]

    # iterate through query results
    for row_id, problem, code, score, in results:
        try:
            # decode source code
            code = code.decode('ASCII')
        except UnicodeDecodeError:
            print(f'error decode at id = {row_id}')
        else:
            # preprocessing includes normalization and tokenization
            problem_processed, comments_processed = preprocess(
                problem, code)

            # create vector of 0s equal to the amount of words in our header template text
            header_word_number = [0 for i in range(len(header_parsed))]
            i = 0
            # loop through header words
            for header_word in header_parsed:
                # loop through comment lines
                for line in comments_processed:
                    # loop through words in a line
                    for comment_word in line:
                        # check if word from comment has < 3 levenshtein distance from the header word
                        #   use levenshtein distance to take account typos, e.g. "keberkahannya vs keberhakanna"
                        if levenshtein_distance(header_word, comment_word) < 3:
                            # if word from comment checks out, mark the 0 from the vector as 1
                            #   indicating that the word exists in the comments from the code
                            header_word_number[i] = 1
                i += 1

            avg_header_score = \
                sum(header_word_number) / \
                len(header_word_number)
            # print(header_parsed)
            # print(comments_raw)
            # print(header_word_number)
            print(f'{row_id} - {avg_header_score}')
            # break
