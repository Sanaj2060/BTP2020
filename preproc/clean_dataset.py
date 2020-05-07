'''
Output should be csv dataset with the conditions:

  1. All rows should exactly contain only two words (lang1 and lang2)
  2. There should not be any invalid tokens
  3. Contains no redundant dataset (i.e. No two rows should have exact same dataset pairs)

  *** Subsequent processing of the dataset assumes the above two conditions are
      satisfied for the and process without any error handling of the above two
      conditions
'''

import pandas as pd
import argparse
import random

def parse_cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='in_file', required=True, help='Dataset file to be cleaned')
    parser.add_argument('-l', dest='lang_code', required=True, help='Language code')

    return parser.parse_args()


# language independant normalisation
def normalise(word):
    word = word.strip()
    word = word.replace('\u200b', '')
    word = word.replace('\u200e', '')
    word = word.replace('\u200f', '')
    word = word.replace('\u200d', '')
    word = word.replace('\u200c', '')  # zero-width non joiner
    word = word.replace('\xad', '')  # remove soft hyphen
   
    return word

# set of invalid tokens
filters = "~!@#$%^&*()-_=+[{]}\|;:\",<>/?1234567890"
filters = set(filters)

# input = [lang1, lang2]
# will validate for condotion 1 and 2
# should be language independant
# if given, valid_tokens should contain all the valid tokens of lang1 and lang2
def valid(dataset_row, length = 32, valid_tokens=None):
    if len(dataset_row) != 2:
        return False

    try:
        dataset_row[0] = normalise(dataset_row[0])
        dataset_row[1] = normalise(dataset_row[1])
    except:
        print('Invalid dataset: {}'.format(dataset_row))
        return False

    if len(dataset_row[0]) > length or len(dataset_row[1]) > length:
        print('Invalid dataset: {}'.format(dataset_row))
        return False

    for tk in dataset_row[0] + dataset_row[1]:
        if tk in filters:
            print('Invalid dataset: {}'.format(dataset_row))
            return False
        if valid_tokens and tk not in valid_tokens:
            print('Invalid dataset: {}'.format(dataset_row))
            return False
    return True

# should be language independant
# type(dataset) = list
def clean_and_write(dataset, out_file):
    invalid_count = 0
    redundant_count = 0
    total = len(dataset)
    unique_dataset = set()

    print('Total input datasets: {}\n'.format(total))

    random.shuffle(dataset)
    for i, dataset_row in enumerate(dataset):
        if valid(dataset_row):
            dataset_row = tuple(dataset_row)
            if dataset_row not in unique_dataset:
                unique_dataset.add(dataset_row)
                out_file.write(','.join(dataset_row) + '\n')
            else:
                print('Redundant: {}'.format(dataset_row))
                redundant_count += 1
        else:
            invalid_count += 1
    
    return total, total -(invalid_count + redundant_count), invalid_count, redundant_count


if __name__ == '__main__':
    cl_args = parse_cl_args()

    out_datast_file_path = 'dataset/{}/{}.csv'.format(cl_args.lang_code, cl_args.lang_code)

    try:
        dataset = pd.read_csv(cl_args.in_file, sep=',', error_bad_lines=False, header=None).values.tolist()
    except Exception as e:
        print(e)
        print('File not exist: ', cl_args.in_file)
        exit()

    try:
        out_dataset_file = open(out_datast_file_path, 'w')
    except:
        print('Cannot open file: ', out_datast_file_path)
        exit()

    total_in, total_out, invalid_count, redundant_count = clean_and_write(dataset, out_dataset_file)

    print('\nInvalid count: ', invalid_count)
    print('Redundant count: ', redundant_count)
    print('\nTotal input dataset: ', total_in)
    print('\nTotal output dataset: ', total_out)
    print('\nCompleted')
