import tensorflow as tf
import os
import shutil
import numpy as np

import param
import data


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

#@tf.function
def predict(inp_sequence, transformer):
    # shape(inp_sequence) = (pad_size)
    encoder_input = tf.expand_dims(inp_sequence, 0)
    
    # the first token to the decoder of the transformer should be the SOS.
    decoder_input = [param.SOS_ID]
    decoder_input = tf.expand_dims(decoder_input, 0)
        
    for i in range(param.PAD_SIZE):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, decoder_input)
    
        # predictions.shape == (batch_size, pad_size, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                    decoder_input,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

        # select the last word from the seq_len dimension
        last_char = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(last_char, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == param.EOS_ID:
            break
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    predictions = tf.squeeze(predictions, axis=0)
    pad = param.PAD_SIZE - tf.shape(predictions).numpy()[0]
    padding = tf.constant([[0, pad], [0, 0]])
    predictions = tf.pad(predictions, padding, 'CONSTANT')
    return predictions # shape(predictions) = (pad_size, tar_vocab_size)

def mk_eval_dataset(dataset):
    eval_dataset = dict()
    for dataset_row in dataset:
        key = tuple(dataset_row[0].numpy())
        if key in eval_dataset:
            eval_dataset[key].append(dataset_row[2])
        else:
            eval_dataset[key] = [dataset_row[2]]
    return eval_dataset

best_path = 'records/active/checkpoints/best'

def get_best():
    if os.path.exists(best_path):
        try:
            with open(best_path, 'r') as file:
                line = file.read()
                best_loss = float(line.split(' ')[0])
                best_checkpoint = line.split(' ')[1]
        except Exception as e:
            print('best file exist but cannot open: {}'.format(e))
            exit()
        return best_loss, best_checkpoint
    else:
        return 999, None

def record_best(best_loss, best_checkpoint):
    if best_checkpoint is None:
        print('best_checkpoint is None, so not saving best')
        return
    try:
        with open(best_path, 'w') as file:
            file.write('{} {}'.format(best_loss, best_checkpoint))
    except:
        print('Cannot open file: records/active/checkpoints/best')

def get_time(secs):
    h = int(secs // (60 * 60))
    rem_sec = secs - (h * 60 * 60)
    m = int(rem_sec // 60)
    s = rem_sec - (m * 60)

    return '{} hrs {} min {:.2f} secs'.format(h, m, s)

class TrainDetails:
    def __init__(self, details_path, extra=None):
        self.details_path = details_path
        self.elapsed_time = 0
        self.param_file = None

        self.create_req_files()
        self.save_params(extra)

    def create_req_files(self):
        try:
            if not os.path.exists(self.details_path):
                os.makedirs(self.details_path)
        except:
            print('TrainDetails directory creation failed')
            sys.exit()

        try:
            if not os.path.exists('{}/time'.format(self.details_path)):
                self.time_file = open('{}/time'.format(self.details_path), 'w')
                self.time_file.write('0')
                print('time file does not exist, created new file')
                self.elapsed_time = 0
            else:
                self.time_file = open('{}/time'.format(self.details_path), 'r+')
                tm = self.time_file.read()
                self.elapsed_time = float(tm)
                print('loaded elapsed_time from     time, total elapsed time is: ', tm, ' secs')
        except:
            print('time file creation failed')
            exit()

        try:
            if not os.path.exists('{}/metric'.format(self.details_path)):
                self.metric_file = open('{}/metric'.format(self.details_path), 'w')
                print('metric file does not exist, created new file')
            else:
                self.metric_file = open('{}/metric'.format(self.details_path), 'a')
        except:
            print('param file creation failed')
            sys.exit()

        try:
            self.param_file = open('{}/param'.format(self.details_path), 'w')
        except:
            print('metric file creation failed')
            sys.exit()
        
    def save_params(self, extra=None):
        if self.param_file is None:
            return
        if extra is not None:
            self.param_file.write('{}\n'.format(extra))
        self.param_file.write('DFF = {}\n'.format(param.DFF))
        self.param_file.write('DROPOUT = {}\n'.format(param.DROPOUT))
        self.param_file.write('D_MODEL = {}\n'.format(param.D_MODEL))
        self.param_file.write('PAD_SIZE = {}\n'.format(param.PAD_SIZE))
        self.param_file.write('NUM_HEADS = {}\n'.format(param.NUM_HEADS))
        self.param_file.write('BATCH_SIZE = {}\n'.format(param.BATCH_SIZE))
        self.param_file.write('NUM_LAYERS = {}\n'.format(param.NUM_LAYERS))
        self.param_file.write('EMB_DIM = D_MODEL')

    def save_elapsed_time(self, tm):
        self.elapsed_time += tm
        self.time_file.seek(0)
        self.time_file.truncate()
        self.time_file.write('{:.4f}'.format(self.elapsed_time))
        return self.elapsed_time

    def save_metric(self, metric):
        self.metric_file.write('{}\n'.format(metric))

    def rm_details_file(self):
        if os.path.exists('{}/time'.format(self.details_path)):
            os.remove('{}/time'.format(self.details_path))
        if os.path.exists('{}/metric'.format(self.details_path)):
            os.remove('{}/metric'.format(self.details_path))


learning_rate = CustomSchedule(param.D_MODEL, warmup_steps=2000)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)