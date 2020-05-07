import numpy as np
import pandas as pd


class Tokenizer:
    def __init__(self, token_file_path, rev=True):
        self.__rev__ = rev
        self.inp_lookup = dict()
        self.tar_lookup = dict()
        self.inp_rev_lookup = dict()
        self.tar_rev_lookup = dict()
        
        self.sos = '%'
        self.eos = '$'
        self.unk = '#'
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.inp_vocab_size = 0
        self.tar_vocab_size = 0

        self.__make_lookup_tables__(token_file_path)


    def __make_lookup_tables__(self, token_file_path):
        if token_file_path.split('.')[-1] != 'tokens':
            print('Unknow tokens file: ', token_file_path)
            exit()
        try:
            with open(token_file_path, 'r') as file:
                tokens = file.read()
        except:
            print('Tokenizer: File not found [ {} ]'.format(token_file_path))
            exit()

        if self.__rev__:
            inp_tokens = tokens.split(',')[1]
            tar_tokens = tokens.split(',')[0]
            print(tar_tokens)
        else:
            inp_tokens = tokens.split(',')[0]
            tar_tokens = tokens.split(',')[1]

        # i + 4: 3 for special tokens 1 for padding
        self.inp_lookup[self.sos] = 1
        self.inp_lookup[self.eos] = 2
        self.inp_lookup[self.unk] = 3
        for i, tk in enumerate(inp_tokens):
            self.inp_lookup[tk] = i + 4
            self.inp_rev_lookup[i + 4] = tk

        self.tar_lookup[self.sos] = 1
        self.tar_lookup[self.eos] = 2
        self.tar_lookup[self.unk] = 3
        for i, tk in enumerate(tar_tokens):
            self.tar_lookup[tk] = i + 4
            self.tar_rev_lookup[i + 4] = tk

        # vocab size += 3, for sos eos and unk
        self.inp_vocab_size = len(self.inp_lookup) + 1
        self.tar_vocab_size = len(self.tar_lookup) + 1

    # encode a single word
    # word is not modified
    # return encoded numpy array
    def inp_encode(self, word, pad_size = None):
        if pad_size:
            en_vec_sz = pad_size
        else:
            en_vec_sz = len(word)

        en_vec = np.zeros(en_vec_sz, dtype='float32')
        for i, ch in enumerate(word):
            en_vec[i] = self.inp_lookup.get(ch, self.unk_id)
        return en_vec
        
    # encode a single word
    # word is not modified
    # return encoded numpy array
    def tar_encode(self, word, pad_size = None):
        if pad_size:
            en_vec_sz = pad_size
        else:
            en_vec_sz = len(word)

        en_vec = np.zeros(en_vec_sz, dtype='float32')
        for i, ch in enumerate(word):
            en_vec[i] = self.tar_lookup.get(ch, self.unk_id)
        return en_vec

    # decode a single encoded numpy array,  return str
    # en_vec is not modified
    # special tokens: sos, eos, unk
    def inp_decode(self, en_vec, decode_special_tokens = False):
        word = ''
        for tk in en_vec:
            if tk == 0 and not decode_special_tokens:
                return word
            if tk == self.sos_id:
              if decode_special_tokens:
                word += self.sos
                continue
              else:
                continue
            if tk == self.eos_id:
                if decode_special_tokens:
                  word += self.eos
                  continue
                else:
                  return word
            word += self.inp_rev_lookup.get(tk, self.unk)

        return word

    # decode a single encoded numpy array,  return str
    # en_vec is not modified
    # special tokens: sos, eos, unk
    def tar_decode(self, en_vec, decode_special_tokens = False):
        word = ''
        for tk in en_vec:
            if tk == self.sos_id:
              if decode_special_tokens:
                word += self.sos
                continue
              else:
                continue
            
            if tk == self.eos_id:
                if decode_special_tokens:
                  word += self.eos
                  continue
                else:
                  return word
            word += self.tar_rev_lookup.get(tk, self.unk)

        return word

    # encode dataset, dataset is a list of input and target pairs, dtype = list([str, str])
    # dataset is not modified
    # return encoded dataset, dtype = np array of float32
    def encode_dataset(self, dataset, pad_size = None):
        en_dataset = []
        for dataset_row in dataset:
            if self.__rev__:
                inp = dataset_row[1]
                tar = dataset_row[0]
            else:
                inp = dataset_row[0]
                tar = dataset_row[1]

            d = []
            d.append(self.inp_encode(self.sos + inp + self.eos, pad_size))
            d.append(self.tar_encode(self.sos + tar, pad_size))
            d.append(self.tar_encode(tar + self.eos, pad_size))
            en_dataset.append(d)

        return np.asarray(en_dataset)
