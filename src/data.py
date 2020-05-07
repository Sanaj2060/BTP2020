import tensorflow as tf
import pandas as pd
import numpy as np

import tokenizer
import param


class Data:
    def __init__(self, lang_code, rev):
        tk_file_path = 'dataset/{}/{}.tokens'.format(lang_code, lang_code)     
        # Tokenizer
        self.tokenizer = tokenizer.Tokenizer(tk_file_path, rev)

        try:
            train_dataset_path = 'dataset/{}/{}-train.csv'.format(lang_code, lang_code)
            self.train_dataset = pd.read_csv(train_dataset_path, header=None).values.tolist()
        except:
            print('Cannot open file: ', train_dataset_path)
            exit()
        try:
            val_dataset_path = 'dataset/{}/{}-val.csv'.format(lang_code, lang_code)
            self.val_dataset = pd.read_csv(val_dataset_path, header=None).values.tolist()
        except:
            print('Cannot open file: ', val_dataset_path)
            exit()
        try:
            test_dataset_path = 'dataset/{}/{}-test.csv'.format(lang_code, lang_code)
            self.test_dataset = pd.read_csv(test_dataset_path, header=None).values.tolist()
        except:
            print('Cannot open file: ', test_dataset_path)
            exit()


        self.inp_vocab_size = self.tokenizer.inp_vocab_size
        self.tar_vocab_size = self.tokenizer.tar_vocab_size

    def get_dataset(self, short_test=False):
        if short_test:
            n = 10
            self.train_dataset = self.train_dataset[:n]
            self.test_dataset  = self.test_dataset[:n]
            self.val_dataset   = self.val_dataset[:n]

        print('\n\ntrain_dataset size: ', len(self.train_dataset))
        print('test_dataset size: ', len(self.test_dataset))
        print('val_dataset size: ', len(self.val_dataset), '\n\n')

        train_dataset = self.tokenizer.encode_dataset(self.train_dataset, param.PAD_SIZE)
        test_dataset = self.tokenizer.encode_dataset(self.test_dataset, param.PAD_SIZE)
        val_dataset = self.tokenizer.encode_dataset(self.val_dataset, param.PAD_SIZE)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.shuffle(60000)
        train_dataset = train_dataset.batch(param.BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)

        return train_dataset, test_dataset, val_dataset

    def get_dataset_size(self):
        return (len(self.train_dataset), len(self.test_dataset), len(self.val_dataset))
