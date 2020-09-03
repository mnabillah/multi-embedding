import csv
from datetime import datetime

from functions import get_data, preprocess

if __name__ == '__main__':
    results = get_data()

    csv_file = open(
        f'density-{datetime.now().year}-{datetime.now().month:02d}-{datetime.now().day:02d}.csv',
        'w',
        newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['id',
                     'comment_line_count', 'code_line_count',
                     'comment_char_count', 'code_char_count',
                     'comment_line_density', 'comment_char_density'])

    for row_id, problem, code, _ in results:
        # there were decoding problems with some code files, so I put it in the try block
        try:
            # decode source code
            code = code.decode('ASCII')
        except UnicodeDecodeError:
            print(f'error decoding blob at row={row_id}')
        else:
            problem_processed, comments_processed, code_only = preprocess(problem, code)
            comment_line_count = len(comments_processed)
            code_line_count = len(code_only)
            comment_char_count = len(''.join([''.join(line) for line in comments_processed]))
            code_char_count = len(''.join(code_only).strip(' '))
            comment_line_density = comment_line_count / (comment_line_count + code_line_count)
            comment_char_density = comment_char_count / (comment_char_count + code_char_count)
            # print(f"Total comment lines      : {comment_line_count}")
            # print(f"Total code lines         : {code_line_count}")
            # print(f"Total comment characters : {comment_char_count}")
            # print(f"Total code characters    : {code_char_count}")
            # print(f"Comment line density     : {comment_line_density}")
            # print(f"Comment char density     : {comment_char_density}")
            # exit(0)
            writer.writerow([row_id,
                             comment_line_count, code_line_count,
                             comment_char_count, code_char_count,
                             comment_line_density, comment_char_density])
