import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', dest='in_file', required=True, help='Dataset file to be cleaned')
parser.add_argument('-l', dest='lang_code', required=True, help='Language code')
cl_args = parser.parse_args()

csv_file_path = cl_args.in_file

lang_code = cl_args.lang_code

tokens_file_path = 'dataset/{}/{}.tokens'.format(lang_code, lang_code)

print('Getting tokens from: ', csv_file_path)
print('Tokens file at: ', tokens_file_path)

try:
    csv_file = pd.read_csv(csv_file_path, header=None)
except Exception as e:
    print(e)
    print('\nCannot open file: ', csv_file_path)
    exit()

try:
    tokens_file = open(tokens_file_path, 'w')
except:
    print('Cannot open file: ', tokens_file_path)
    exit()


lang1_tokens = set()
lang2_tokens = set()


print('\n\tGetting tokens ...')
for dataset in csv_file.values.tolist():
    lang1 = dataset[0]
    lang2 = dataset[1]

    for tk in lang1:
        lang1_tokens.add(tk)
    for tk in lang2:
        lang2_tokens.add(tk)

lang1_tokens = sorted(list(lang1_tokens))
lang2_tokens = sorted(list(lang2_tokens))

print('\nNumber of lang1 tokens: ', len(lang1_tokens))
print('Number of lang2 tokens: ', len(lang2_tokens))


for tk in lang1_tokens:
    tokens_file.write(tk)
tokens_file.write(',')
for tk in lang2_tokens:
    tokens_file.write(tk)

print('\nCompleted')
